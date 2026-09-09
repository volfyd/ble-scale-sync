import type {
  BleDeviceInfo,
  CharacteristicBinding,
  ConnectionContext,
  ScaleAdapterCore,
  GattWiring,
  HoldForComposition,
  MultiCharNotify,
  ScaleReading,
  UserProfile,
  BodyComposition,
} from '../interfaces/scale-adapter.js';
import { uuid16, buildPayload, type ScaleBodyComp } from './body-comp-helpers.js';
import { matchesDescriptor, type MatchDescriptor } from './match-descriptor.js';
import { bleLog, errMsg } from '../ble/types.js';

// Original Trisa firmware exposes 0x8A21 (notify) for measurement.
const CHR_MEASUREMENT_TRISA = uuid16(0x8a21);
// ADE BA 1600 / fitvigo firmware does NOT expose 0x8A21. Measurement frames
// arrive on 0x8A24 (indicate) instead. Frame layout (weight portion) is
// compatible with the Trisa decoder; body composition encoding still TBD.
const CHR_MEASUREMENT_ADE = uuid16(0x8a24);
// On ADE the scale also pushes another payload on 0x8A22 (indicate) shortly
// after the weight frame. Encoding is not yet decoded; we subscribe to it
// purely so future captures with debug logging can collect the bytes.
const CHR_BODYCOMP_ADE = uuid16(0x8a22);
// 0x8A82 is the upload channel on both variants. Trisa sends password (0xA0)
// + challenge (0xA1); ADE only sends challenge (0xA1) without a preceding
// password frame.
const CHR_UPLOAD = uuid16(0x8a82);
// 0x8A81 is the host -> scale write channel on both variants.
const CHR_DOWNLOAD = uuid16(0x8a81);

// openScale opcodes for the Trisa challenge-response protocol.
const OP_PASSWORD = 0xa0;
const OP_CHALLENGE = 0xa1;
// Time-sync command: opcode 0x02 followed by 4-byte LE seconds-since-2010.
// Identical on Trisa and ADE.
const OP_TIME_SYNC = 0x02;
// Final "pairing complete" / broadcast-id opcode. Trisa uses 0x21, ADE 0x22.
const OP_BROADCAST_TRISA = 0x21;
const OP_BROADCAST_ADE = 0x22;
// Challenge response opcode. Trisa echoes 0xA1; ADE uses 0x20 (the response
// payload encoding is also different; see handleUploadChannel).
const OP_RESPONSE_TRISA = 0xa1;
const OP_RESPONSE_ADE = 0x20;

// Weight Gurus A3 (Transtek) opcodes — same service as Trisa/ADE but
// different handshake sequence and response opcodes.
const OP_WG_ACCOUNT_ID = 0x21;
const OP_WG_VERIFICATION = 0x20;
const OP_WG_ENABLE_DISCONNECT = 0x22;
const OP_WG_ADD_USER = 0x03;
const OP_WG_PROFILE = 0x51;
const OP_WG_SLOT_STATUS = 0x83;
const WG_LAST_SLOT = 8;
const WG_SLOT_NAME_LEN = 18;

const EPOCH_2010 = 1262304000;

function decodeSfloat16(raw: number): number {
  let exponent = (raw >> 12) & 0x0f;
  if (exponent >= 8) exponent -= 16;
  let mantissa = raw & 0x0fff;
  if (mantissa >= 0x0800) mantissa -= 0x1000;
  return mantissa * Math.pow(10, exponent);
}

/**
 * Retry budget for a challenge-response write. BlueZ answers a badly timed
 * write with a transient `org.bluez.Error.InProgress`, and the scale is waiting
 * on that single ack (#138).
 */
const CHALLENGE_WRITE_RETRIES = 2;
const CHALLENGE_RETRY_MS = 250;

type Variant = 'trisa' | 'ade' | 'weightgurus';

/**
 * Adapter for the Trisa body-composition scale family.
 *
 * Three firmware variants are supported:
 *   - Trisa (default): exposes 0x8A21 (notify) for measurement, full
 *     password + challenge handshake on 0x8A82.
 *   - ADE BA 1600 / fitvigo: 0x8A21 is missing; measurement arrives on 0x8A24
 *     (indicate). Different challenge-response and different
 *     "pairing complete" opcode (0x22 instead of 0x21). Body-composition
 *     decoding is not yet implemented; only weight is reported.
 *   - Weight Gurus A3 (0375/0376, Transtek): uses 0x8A24 for measurement like
 *     ADE, but speaks a multi-step Trisa-like handshake with different opcodes
 *     (0x20 verification, 0x21 account ID) and requires waiting for 8 slot
 *     status frames before completing setup.
 *
 * Variant detection: if a 0xA0 password frame arrives before onConnected and
 * 0x8A21 is absent → weightgurus. Otherwise the original char-based detection
 * applies.
 */
export class TrisaAdapter implements ScaleAdapterCore, GattWiring, MultiCharNotify, HoldForComposition {
  readonly name = 'Trisa';
  readonly match: MatchDescriptor = { priority: 140, names: { startsWith: ['01257b', '11257b', '1202b', '0202b'] } };
  // Legacy single-char fallback (only used when `characteristics` is ignored).
  readonly charNotifyUuid = CHR_MEASUREMENT_TRISA;
  readonly charWriteUuid = CHR_DOWNLOAD;

  readonly normalizesWeight = true;

  readonly characteristics: CharacteristicBinding[] = [
    // Trisa-only measurement char.
    { uuid: CHR_MEASUREMENT_TRISA, type: 'notify', optional: true },
    // ADE-only measurement char.
    { uuid: CHR_MEASUREMENT_ADE, type: 'notify', optional: true },
    // ADE-only body-composition push (encoding TBD; captured via debug log).
    { uuid: CHR_BODYCOMP_ADE, type: 'notify', optional: true },
    // Shared upload channel (password + challenge).
    { uuid: CHR_UPLOAD, type: 'notify' },
    // Shared write channel.
    { uuid: CHR_DOWNLOAD, type: 'write' },
  ];

  /** Detected firmware variant. Set in onConnected(). */
  private variant: Variant = 'trisa';
  /** Stored password from opcode 0xA0 (Trisa). ADE does not send this. */
  private password: Buffer | null = null;
  /** Reference to write function, saved from onConnected context. */
  private writeFn: ConnectionContext['write'] | null = null;
  /** Challenge frame that arrived before the connection was ready (#138). */
  private pendingChallenge: Buffer | null = null;
  /**
   * True once onConnected() has run for the current session.
   *
   * The adapter is a shared singleton, so `writeFn` alone cannot say whether a
   * connection is live: it still holds the previous session's closure after a
   * disconnect, and writing through that one throws.
   */
  private connected = false;

  // --- Weight Gurus state ---
  private wgSlotCount = 0;
  private wgSetupDone = false;
  private wgProfile: UserProfile | null = null;
  private isWeightGurusName = false;
  private wgPairingSession = false;
  private wgComposition: ScaleBodyComp | null = null;
  private wgExpectComposition = false;
  private wgLastWeight = 0;
  private wgLastImpedance = 0;
  private wgBmr: number | null = null;

  readonly completionHoldMs = 10_000;

  isFinal(reading: ScaleReading): boolean {
    if (this.variant !== 'weightgurus') return true;
    return this.wgComposition !== null;
  }

  matches(device: BleDeviceInfo): boolean {
    const matched = matchesDescriptor(device, this.match);
    if (matched) {
      const name = (device.localName || '').toLowerCase();
      this.isWeightGurusName = name.startsWith('1202b') || name.startsWith('0202b');
    }
    return matched;
  }

  async onConnected(ctx: ConnectionContext): Promise<void> {
    // NOTE: pendingChallenge is deliberately NOT cleared here. The frame it
    // holds arrived moments ago, before this method ran, and replaying it is the
    // whole point (#138). Stale state from an earlier session is cleared in
    // onSessionEnd() instead, which is where a session actually ends.
    this.writeFn = ctx.write;
    this.connected = true;

    // Both measurement chars are declared `optional` so that variant detection
    // can pick whichever one the firmware exposes. If neither shows up, that
    // is almost certainly a transient GATT discovery race (BlueZ
    // ServicesResolved firing before all chars are exported (bluez/bluez#1489,
    // or the noble equivalent on Windows/macOS). Fail fast with a clear
    // message instead of silently subscribing to no measurement char and
    // stalling on read.
    const hasMeasurement =
      ctx.availableChars.has(CHR_MEASUREMENT_TRISA) || ctx.availableChars.has(CHR_MEASUREMENT_ADE);
    if (!hasMeasurement) {
      throw new Error(
        'Trisa: no measurement characteristic discovered (expected 0x8A21 or 0x8A24). ' +
          'Likely a transient GATT discovery race. Try again.',
      );
    }

    this.variant = this.detectVariant(ctx.availableChars);
    bleLog.debug(`Trisa adapter: variant=${this.variant}`);

    if (this.variant === 'weightgurus') {
      this.wgSlotCount = 0;
      this.wgSetupDone = false;
      this.wgPairingSession = false;
      this.wgProfile = ctx.profile;

      // On established sessions the scale doesn't resend 0xA0, so derive the
      // password from the BLE MAC: first 4 octets in reverse byte order.
      if (!this.password && ctx.deviceAddress) {
        const mac = ctx.deviceAddress.replace(/:/g, '');
        if (mac.length >= 8) {
          const b = Buffer.from(mac.slice(0, 8), 'hex');
          this.password = Buffer.from([b[3]!, b[2]!, b[1]!, b[0]!]);
          bleLog.debug(`WG password derived from MAC: ${this.password.toString('hex')}`);
        }
      }

      // The 0xA0 password frame typically arrives before onConnected(), so the
      // Trisa path stores the password but can't write (no writeFn yet). Now
      // that we have a write function and know we're weightgurus, send the
      // account ID that the scale is waiting for.
      if (this.password && this.pendingChallenge === null) {
        // Pairing session: password received via 0xA0, no queued challenge yet.
        bleLog.debug('WG password already received, sending account ID');
        const accountId = Buffer.alloc(4);
        accountId.writeUInt32LE((Math.floor(Math.random() * 0x7ffffffe) + 1) >>> 0, 0);
        await ctx.write(CHR_DOWNLOAD, Buffer.from([OP_WG_ACCOUNT_ID, ...accountId]), true);
        bleLog.debug('WG account ID sent');
      }

      const queued: Buffer | null = this.pendingChallenge;
      if (queued) {
        this.pendingChallenge = null;
        bleLog.debug(`Replaying queued frame: ${queued.toString('hex')}`);
        this.handleUploadChannel(queued);
      }
      return;
    }

    // Time sync (same opcode on both variants).
    const now = Math.floor(Date.now() / 1000) - EPOCH_2010;
    const tsCmd = Buffer.alloc(5);
    tsCmd[0] = OP_TIME_SYNC;
    tsCmd.writeUInt32LE(now, 1);
    await ctx.write(CHR_DOWNLOAD, [...tsCmd], true);

    // Broadcast / pairing-complete opcode differs between variants.
    const broadcastOp = this.variant === 'ade' ? OP_BROADCAST_ADE : OP_BROADCAST_TRISA;
    await ctx.write(CHR_DOWNLOAD, [broadcastOp], true);

    // Replay a challenge that beat this method to the notification handler, now
    // that both the write function and the variant are known (#138).
    const queued: Buffer | null = this.pendingChallenge;
    if (queued) {
      this.pendingChallenge = null;
      bleLog.debug(`Replaying queued challenge: ${queued.toString('hex')}`);
      this.handleUploadChannel(queued);
    }
  }

  /**
   * End of a GATT session: forget everything tied to it.
   *
   * `writeFn` is a closure over the connection that just went away, so anything
   * written through it after this point is discarded by the transport (on the
   * mqtt-proxy it is published into the void and even resolves successfully).
   * Clearing `connected` is also what re-arms the pre-connect challenge queue
   * for the next cycle; without it the #138 fix would work exactly once per
   * process, which in continuous mode means once ever.
   */
  onSessionEnd(): void {
    this.connected = false;
    this.writeFn = null;
    this.pendingChallenge = null;
    // Preserve password and variant across sessions — the adapter is a
    // singleton and established sessions don't resend 0xA0.
    this.wgSlotCount = 0;
    this.wgSetupDone = false;
    this.wgProfile = null;
    this.wgExpectComposition = false;
    this.wgLastWeight = 0;
  }

  /**
   * Variant precedence: pick `weightgurus` when a password frame was received
   * (Trisa-style auth) but 0x8A21 is absent (ADE-style measurement). Pick
   * `ade` only when no password and 0x8A21 absent.
   */
  private detectVariant(available: ReadonlySet<string>): Variant {
    const hasTrisa = available.has(CHR_MEASUREMENT_TRISA);
    const hasAde = available.has(CHR_MEASUREMENT_ADE);
    if (this.isWeightGurusName && !hasTrisa && hasAde) return 'weightgurus';
    if (this.password && !hasTrisa && hasAde) return 'weightgurus';
    if (this.password) return 'trisa';
    if (!hasTrisa && hasAde) return 'ade';
    return 'trisa';
  }

  /**
   * Dispatch notifications from different characteristics.
   *
   * Trisa:
   *   - 0x8A82: password (0xA0) and challenge (0xA1) frames
   *   - 0x8A21: measurement data
   * ADE BA 1600:
   *   - 0x8A82: challenge (0xA1), no password frame; response algo unknown
   *   - 0x8A24: measurement data (Trisa-compatible weight encoding)
   *   - 0x8A22: body-composition push (encoding TBD)
   * Weight Gurus A3:
   *   - 0x8A82: password (0xA0), challenge (0xA1), slot status (0x83)
   *   - 0x8A24: weight measurement (32-bit IEEE-11073 FLOAT)
   *   - 0x8A22: body composition (16-bit SFLOATs)
   */
  parseCharNotification(charUuid: string, data: Buffer): ScaleReading | null {
    if (charUuid === CHR_UPLOAD) {
      this.handleUploadChannel(data);
      return null;
    }
    if (charUuid === CHR_MEASUREMENT_TRISA || charUuid === CHR_MEASUREMENT_ADE) {
      return this.parseMeasurement(data);
    }
    if (charUuid === CHR_BODYCOMP_ADE) {
      if (this.variant === 'weightgurus') {
        return this.parseWgBodyComp(data);
      }
      bleLog.debug(`Body-comp frame on 0x8A22: ${data.toString('hex')}`);
      return null;
    }
    return null;
  }

  /**
   * Fallback for legacy single-char path. Parses measurement data only.
   */
  parseNotification(data: Buffer): ScaleReading | null {
    return this.parseMeasurement(data);
  }

  isComplete(reading: ScaleReading): boolean {
    return reading.weight > 0;
  }

  computeMetrics(reading: ScaleReading, profile: UserProfile): BodyComposition {
    const comp: ScaleBodyComp = this.wgComposition ?? {};
    const result = buildPayload(reading.weight, reading.impedance, comp, profile);
    if (this.wgBmr != null) result.bmr = this.wgBmr;
    return result;
  }

  /**
   * Handle password and challenge frames from the upload channel (0x8A82).
   *
   * Trisa flow: scale sends 0xA0 (password), then 0xA1 (challenge); host
   * responds with [0xA1, XOR(challenge, password)].
   *
   * ADE BA 1600 flow: scale sends 0xA1 (challenge) directly without a
   * password frame. fitvigo's native protocol (`corelib::VBaseA2PairingProtocol`
   * + `ProtocolUtils::sendVerificationCode`) computes the response as
   * `[0x20, LE32(savedPassword XOR challengeInt)]`, where `challengeInt` is
   * the four bytes after the opcode read as little-endian uint32. Because
   * BE1615 never receives a 0xA0 frame, `savedPassword` stays at its default
   * zero, so the response collapses to `[0x20]` followed by an echo of the
   * same four bytes.
   *
   * Weight Gurus A3 flow:
   *   Pairing: 0xA0 → send 0x21 (account ID); 0xA1 → send 0x20 (verification
   *   = XOR(pw, challenge)); then 8x 0x83 → send 0x03+0x51+0x02+0x22.
   *   Established: 0xA1 → send 0x20, 0x02 (time), 0x51 (profile).
   */
  private handleUploadChannel(data: Buffer): void {
    if (data.length < 2) return;
    const opcode = data[0];

    // On Linux the CCCD write and onConnected() run in parallel, so the scale's
    // challenge can arrive before this adapter has a connection at all. Neither
    // the write function nor the detected variant exists yet, and the frame used
    // to be dropped without so much as a log line: the scale then waited for an
    // ack that never came, once per cycle, forever (#138). Hold the raw frame
    // and replay it from onConnected(), where both are known.
    if ((opcode === OP_CHALLENGE || opcode === OP_WG_SLOT_STATUS) && !this.connected) {
      this.pendingChallenge = Buffer.from(data);
      bleLog.debug('Frame arrived before connect completed; queued for replay');
      return;
    }

    if (this.variant === 'weightgurus') {
      this.handleWeightGurusUpload(opcode, data);
      return;
    }

    if (this.variant === 'ade') {
      if (opcode === OP_CHALLENGE && data.length >= 5) {
        // Echo the four bytes that follow the opcode (XOR with savedPassword=0).
        const response = Buffer.from([OP_RESPONSE_ADE, data[1], data[2], data[3], data[4]]);
        this.sendResponse(
          response,
          `ADE challenge ack sent: ${response.toString('hex')}`,
          Buffer.from(data),
        );
      } else {
        bleLog.debug(`ADE upload frame (unhandled opcode): ${data.toString('hex')}`);
      }
      return;
    }

    if (opcode === OP_PASSWORD) {
      this.password = Buffer.from(data.subarray(1));
    } else if (opcode === OP_CHALLENGE && this.password) {
      const challenge = data.subarray(1);
      const response = Buffer.alloc(challenge.length + 1);
      response[0] = OP_RESPONSE_TRISA;
      for (let i = 0; i < challenge.length; i++) {
        response[i + 1] = challenge[i] ^ (this.password[i % this.password.length] ?? 0);
      }
      this.sendResponse(response, undefined, Buffer.from(data));
    }
  }

  private handleWeightGurusUpload(opcode: number, data: Buffer): void {
    if (opcode === OP_PASSWORD) {
      if (data.length < 5) return;
      this.password = Buffer.from(data.subarray(1, 5));
      this.wgPairingSession = true;
      bleLog.debug(`WG password received: ${this.password.toString('hex')}`);

      // Pairing: claim an account ID so the scale commits the pairing.
      const accountId = Buffer.alloc(4);
      accountId.writeUInt32LE((Math.floor(Math.random() * 0x7ffffffe) + 1) >>> 0, 0);
      this.sendResponse(
        Buffer.from([OP_WG_ACCOUNT_ID, ...accountId]),
        `WG account ID sent`,
      );
      return;
    }

    if (opcode === OP_CHALLENGE) {
      if (data.length < 5) return;
      const pw = this.password;
      if (!pw) {
        bleLog.debug('WG challenge without password — scale not paired');
        return;
      }
      const challenge = data.subarray(1, 5);
      const xored = Buffer.alloc(4);
      for (let i = 0; i < 4; i++) {
        xored[i] = challenge[i]! ^ (pw[i % pw.length] ?? 0);
      }
      bleLog.debug(`WG challenge: ${challenge.toString('hex')} → verification: ${xored.toString('hex')}`);
      const onVerified = !this.wgPairingSession
        ? () => {
            bleLog.debug('WG established session, sending time + profile');
            this.wgSendTimeAndProfile(1);
          }
        : undefined;

      this.sendResponse(
        Buffer.from([OP_WG_VERIFICATION, ...xored]),
        `WG verification sent`,
        Buffer.from(data),
        onVerified,
      );
      // Pairing session: wait for 8x 0x83 slot status frames before setup.
      return;
    }

    if (opcode === OP_WG_SLOT_STATUS) {
      if (data.length < 2) return;
      const slot = data[1] & 0xff;
      this.wgSlotCount++;
      bleLog.debug(`WG slot ${slot} status (${this.wgSlotCount}/${WG_LAST_SLOT}): ${data.toString('hex')}`);

      if (this.wgSlotCount >= WG_LAST_SLOT && !this.wgSetupDone) {
        bleLog.debug('WG all slots received, completing setup');
        this.wgFinishSetup(1);
      }
      return;
    }

    bleLog.debug(`WG upload frame (opcode 0x${opcode.toString(16)}): ${data.toString('hex')}`);
  }

  private wgFinishSetup(slot: number): void {
    if (this.wgSetupDone) return;
    this.wgSetupDone = true;
    const write = this.writeFn;
    if (!write) return;

    // Add user: [0x03, slot, 18-byte ASCII name padded with spaces]
    const name = Buffer.alloc(WG_SLOT_NAME_LEN, 0x20);
    const userName = 'BLEScaleSync';
    for (let i = 0; i < Math.min(userName.length, WG_SLOT_NAME_LEN); i++) {
      name[i] = userName.charCodeAt(i);
    }
    const addUser = Buffer.from([OP_WG_ADD_USER, slot, ...name]);

    // Profile: [0x51, mask, slot, gender, age, heightLo, heightHi, unit]
    const profile = this.wgProfile;
    const heightCm = profile ? Math.min(profile.height, 204) : 170;
    const heightSfloat = (heightCm * 10) | 0xd000;
    const age = profile ? profile.age : 30;
    const gender = profile?.gender === 'female' ? 0x02 : 0x01;
    const profileCmd = Buffer.from([
      OP_WG_PROFILE,
      0x17, // field mask
      slot,
      gender,
      age & 0xff,
      heightSfloat & 0xff,
      (heightSfloat >> 8) & 0xff,
      0x00, // unit: kg
    ]);

    // Time sync
    const now = Math.floor(Date.now() / 1000) - EPOCH_2010;
    const timeCmd = Buffer.alloc(5);
    timeCmd[0] = OP_TIME_SYNC;
    timeCmd.writeUInt32LE(now, 1);

    // Send the setup sequence. Each write must complete before the next.
    write(CHR_DOWNLOAD, addUser, true)
      .then(() => {
        bleLog.debug('WG add-user sent');
        return write(CHR_DOWNLOAD, profileCmd, true);
      })
      .then(() => {
        bleLog.debug('WG profile sent');
        return write(CHR_DOWNLOAD, timeCmd, true);
      })
      .then(() => {
        bleLog.debug('WG time sync sent');
        return write(CHR_DOWNLOAD, Buffer.from([OP_WG_ENABLE_DISCONNECT]), true);
      })
      .then(() => {
        bleLog.debug('WG enable-disconnect sent — setup complete');
      })
      .catch((err: unknown) => {
        bleLog.debug(`WG setup write failed: ${errMsg(err)}`);
      });
  }

  private wgSendTimeAndProfile(slot: number): void {
    const write = this.writeFn;
    if (!write) return;

    const now = Math.floor(Date.now() / 1000) - EPOCH_2010;
    const timeCmd = Buffer.alloc(5);
    timeCmd[0] = OP_TIME_SYNC;
    timeCmd.writeUInt32LE(now, 1);

    const profile = this.wgProfile;
    const heightCm = profile ? Math.min(profile.height, 204) : 170;
    const heightSfloat = (heightCm * 10) | 0xd000;
    const age = profile ? profile.age : 30;
    const gender = profile?.gender === 'female' ? 0x02 : 0x01;
    const profileCmd = Buffer.from([
      OP_WG_PROFILE,
      0x17,
      slot,
      gender,
      age & 0xff,
      heightSfloat & 0xff,
      (heightSfloat >> 8) & 0xff,
      0x00,
    ]);

    write(CHR_DOWNLOAD, timeCmd, true)
      .then(() => {
        bleLog.debug('WG time sync sent');
        return write(CHR_DOWNLOAD, profileCmd, true);
      })
      .then(() => {
        bleLog.debug('WG profile sent — ready for measurement');
      })
      .catch((err: unknown) => {
        bleLog.debug(`WG established session write failed: ${errMsg(err)}`);
      });
  }

  private parseWgBodyComp(data: Buffer): ScaleReading | null {
    if (data.length < 6) return null;
    const flags = data[0]!;
    let off = 1;

    // Timestamp (uint32 LE) — always present after flags.
    off += 4;

    bleLog.debug(`WG body-comp frame: flags=0x${flags.toString(16)} raw=${data.toString('hex')}`);

    const comp: ScaleBodyComp = {};

    if (flags & 0x01) off += 1; // user id
    if (flags & 0x02) {
      if (off + 2 <= data.length) {
        this.wgBmr = data.readUInt16LE(off);
        bleLog.debug(`WG BMR: ${this.wgBmr} kcal`);
      }
      off += 2;
    }
    if (flags & 0x04) {
      if (off + 2 <= data.length) {
        comp.fat = decodeSfloat16(data.readUInt16LE(off));
        bleLog.debug(`WG body fat: ${comp.fat}%`);
      }
      off += 2;
    }
    if (flags & 0x08) {
      if (off + 2 <= data.length) {
        comp.water = decodeSfloat16(data.readUInt16LE(off));
        bleLog.debug(`WG body water: ${comp.water}%`);
      }
      off += 2;
    }
    if (flags & 0x10) {
      if (off + 2 <= data.length) {
        comp.visceralFat = decodeSfloat16(data.readUInt16LE(off));
        bleLog.debug(`WG visceral fat: ${comp.visceralFat}`);
      }
      off += 2;
    }
    if (flags & 0x20) {
      if (off + 2 <= data.length) {
        comp.muscle = decodeSfloat16(data.readUInt16LE(off));
        bleLog.debug(`WG muscle: ${comp.muscle}%`);
      }
      off += 2;
    }
    if (flags & 0x40) {
      if (off + 2 <= data.length) {
        const bonePercent = decodeSfloat16(data.readUInt16LE(off));
        comp.bone = (bonePercent / 100) * this.wgLastWeight;
        bleLog.debug(`WG bone: ${bonePercent}% → ${comp.bone.toFixed(2)}kg`);
      }
      off += 2;
    }

    this.wgComposition = comp;
    if (this.wgLastWeight > 0) {
      return { weight: this.wgLastWeight, impedance: this.wgLastImpedance };
    }
    return null;
  }

  /**
   * Write a challenge response without awaiting it, but WITH a rejection
   * handler.
   *
   * A bare `void writeFn(...)` here is what took the whole process down in
   * #138: BlueZ answered the write with `org.bluez.Error.InProgress`, nothing
   * was attached to the rejected promise, and Node killed the app on the
   * unhandled rejection. In continuous mode that ends the service; only a
   * container restart policy brought it back.
   */
  private sendResponse(response: Buffer, successLog?: string, replayOnFailure?: Buffer, onSuccess?: () => void): void {
    const write = this.writeFn;
    if (!write) return;
    const attempt = (retriesLeft: number): void => {
      write(CHR_DOWNLOAD, response, true).then(
        () => {
          if (successLog) bleLog.debug(successLog);
          if (onSuccess) onSuccess();
        },
        (err: unknown) => {
          // `org.bluez.Error.InProgress` is what #138 actually hit, and it is
          // transient: another D-Bus operation was in flight on the same
          // adapter. Retrying in-session is what answers the scale on THIS
          // link; queueing for the next one answers a nonce it has forgotten.
          if (retriesLeft > 0 && this.connected) {
            bleLog.debug(`Challenge response write failed, retrying: ${errMsg(err)}`);
            setTimeout(() => attempt(retriesLeft - 1), CHALLENGE_RETRY_MS);
            return;
          }
          bleLog.debug(`Challenge response write failed: ${errMsg(err)}`);
          // Out of retries: the link is most likely already gone. Keep the frame
          // only while no newer one is waiting, so a late failure cannot clobber
          // a fresher challenge.
          if (replayOnFailure && !this.pendingChallenge) this.pendingChallenge = replayOnFailure;
        },
      );
    };
    attempt(CHALLENGE_WRITE_RETRIES);
  }

  /**
   * Parse a Trisa measurement frame.
   *
   * Layout (verified for Trisa 0x8A21 and ADE BA 1600 0x8A24, weight only):
   *   [0]      info flags
   *             bit 0: timestamp present (7 bytes at offset 5)
   *             bit 1: resistance1 present (4 bytes base-10 float)
   *             bit 2: resistance2 present (4 bytes base-10 float)
   *   [1-3]    weight mantissa, unsigned 24-bit little-endian
   *   [4]      weight exponent, signed int8
   *   [5+]     optional timestamp (7 bytes if bit0 set)
   *   then:    optional resistance1 (4 bytes if bit1 set)
   *   then:    optional resistance2 (4 bytes if bit2 set)
   *
   * Weight = mantissa * 10^exponent.
   * Impedance from resistance2: r2 < 410 ? 3.0 : 0.3 * (r2 - 400).
   *
   * Weight Gurus A3 0x8A24 uses the same 32-bit IEEE-11073 FLOAT for weight
   * (bytes 1-4 are identical encoding) but different flag meanings after byte 5:
   *   0x01: timestamp (uint32, 4 bytes — not 7)
   *   0x02: weight delta (32-bit FLOAT, 4 bytes)
   *   0x04: impedance (32-bit FLOAT, 4 bytes)
   *   0x08: user id (uint8)
   *   0x10: status (uint8; bit4 = composition frame follows on 0x8A22)
   *
   * NOTE: only the Trisa branch walks the optional-field table. For ADE the
   * post-weight layout is unverified (timestamp may be 8 bytes instead of 7)
   * and body comp arrives on a separate 0x8A22 push, so the parser
   * short-circuits to weight-only after computing the weight.
   */
  private parseMeasurement(data: Buffer): ScaleReading | null {
    if (data.length < 5) return null;

    const flags = data[0];
    const hasTimestamp = (flags & 0x01) !== 0;
    const hasResistance1 = (flags & 0x02) !== 0;
    const hasResistance2 = (flags & 0x04) !== 0;

    // Skip frames that are just timestamps (only bit0 set, no weight data expected)
    if (hasTimestamp && !hasResistance1 && !hasResistance2) {
      const mantissa = data[1] | (data[2] << 8) | (data[3] << 16);
      if (mantissa === 0) return null;
    }

    // Weight: 24-bit unsigned LE mantissa + signed exponent
    const mantissa = data[1] | (data[2] << 8) | (data[3] << 16);
    const exponent = data.readInt8(4);
    const weight = mantissa * Math.pow(10, exponent);

    if (weight <= 0 || !Number.isFinite(weight)) return null;

    // Weight Gurus A3: the flag layout after the weight is different from Trisa.
    // For now, return weight-only; impedance arrives on the separate 0x8A22
    // body-composition frame which is logged but not yet decoded.
    if (this.variant === 'weightgurus') {
      this.wgLastWeight = weight;
      this.wgComposition = null;
      this.wgExpectComposition = false;
      this.wgBmr = null;

      let off = 5;
      let wgImpedance = 0;
      if (flags & 0x01) off += 4; // timestamp
      if (flags & 0x02) off += 4; // weight delta
      if (flags & 0x04) {
        if (off + 4 <= data.length) {
          const m = data[off]! | (data[off + 1]! << 8) | (data[off + 2]! << 16);
          const e = data.readInt8(off + 3);
          wgImpedance = m * Math.pow(10, e);
          if (wgImpedance > 0) bleLog.debug(`WG impedance: ${wgImpedance.toFixed(1)} ohm`);
        }
        off += 4;
      }
      if (flags & 0x08) off += 1; // user id
      if ((flags & 0x10) && off < data.length) {
        const status = data[off]!;
        this.wgExpectComposition = (status & 0x10) !== 0;
        bleLog.debug(`WG status=0x${status.toString(16)} expectComposition=${this.wgExpectComposition}`);
      }

      this.wgLastImpedance = wgImpedance;
      bleLog.debug(`WG weight frame: ${weight.toFixed(2)}kg impedance=${wgImpedance.toFixed(1)} flags=0x${flags.toString(16)} raw=${data.toString('hex')}`);
      return { weight, impedance: wgImpedance };
    }

    // ADE BA 1600: only the weight bytes are verified (single capture frame in
    // #138). The post-weight layout (timestamp width, resistance encoding)
    // is not confirmed and body-comp values arrive on a separate 0x8A22 push
    // anyway. Don't walk the offset table; return weight only until more
    // captures are available.
    if (this.variant === 'ade') return { weight, impedance: 0 };

    // Walk through optional fields to find resistance2.
    let offset = 5;
    if (hasTimestamp) offset += 7;
    if (hasResistance1) offset += 4;

    let impedance = 0;
    if (hasResistance2 && offset + 4 <= data.length) {
      const r2Mantissa = data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16);
      const r2Exponent = data.readInt8(offset + 3);
      const r2 = r2Mantissa * Math.pow(10, r2Exponent);

      if (r2 < 410) {
        impedance = 3.0;
      } else {
        impedance = 0.3 * (r2 - 400);
      }
    }

    return { weight, impedance };
  }
}
