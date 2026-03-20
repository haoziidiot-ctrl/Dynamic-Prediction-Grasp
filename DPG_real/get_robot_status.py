import socket
import struct
import time
from typing import Optional

import numpy as np

# ================= 配置区域 =================
# 请根据机器人实际情况修改 IP
ROBOT_IP = "192.168.100.178"  
ROBOT_PORT = 17804          # 导航控制服务端口

# 协议授权码（16字节）：你的实车配置
# AUTH_CODE = bytes.fromhex("31 04 25 49 2b 32 af 48 8f f1 82 ee 8b 93 5e 68")  #2d
AUTH_CODE = bytes.fromhex("de c9 9d 62 0a 8a f9 4a a4 28 f4 0e fc 46 11 36")  #3d

# 轮询周期（这里默认 20ms）
POLL_PERIOD_S = 0.02

# 仿真坐标系定义:
# 使用「起点 -> 终点」方向作为仿真 x 轴正方向。
# 这两个点请填在与你 UDP 位置一致的平面坐标系下（通常是地图坐标，单位 m）。
SIM_FRAME_START_XY = (0.0, 0.0)
SIM_FRAME_END_XY = (1.0, 0.0)

# 仿真目标点（小球）在 UDP 坐标系下的位置（x, y, z）
# 其中 z 不参与平面旋转，仅作为仿真中的目标高度直接使用。
SIM_TARGET_POS_REAL = (0.25, 0.50, 1.20)

# UDP 接收超时
SOCKET_TIMEOUT_S = 2.0
# ===========================================

_AUTH_SIZE = 16
_HEADER_SIZE = 12
_HEADER_FMT = "<BBHBBBBHH"
_SERVICE_CODE = 0x10
_MAX_RECV_BYTES = 4096

INIT_CONFIDENCE_THRESHOLD = 0.95
INIT_QUERY_PERIOD_S = 0.1


# 0x16 默认示例：按路径点 ID 导航
DEFAULT_NAV_MODE = 1
DEFAULT_NAV_TARGET_POINT_ID = "4"
DEFAULT_NAV_PATH_POINT_IDS = [3, 4]

# 0x14 手动定位（可按需改/或通过命令行传入）
DEFAULT_MANUAL_LOCALIZE_POSE = (0.05, 0.038, 0.0)


def _frame_start_end_xy() -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(SIM_FRAME_START_XY, dtype=float).reshape(2)
    end = np.asarray(SIM_FRAME_END_XY, dtype=float).reshape(2)
    if float(np.linalg.norm(end - start)) < 1e-9:
        raise ValueError("SIM_FRAME_START_XY 与 SIM_FRAME_END_XY 不能重合")
    return start, end


def frame_axes_xy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回:
        start_xy: 新坐标系原点（对应起点）
        x_axis: 新坐标系 x 轴单位向量（起点->终点方向）
        y_axis: 新坐标系 y 轴单位向量（z 轴朝上时左手侧法向）
    """
    start, end = _frame_start_end_xy()
    x_axis = end - start
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-12)
    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=float)
    return start, x_axis, y_axis


def real_xy_to_new_frame_xy(x_real: float, y_real: float) -> np.ndarray:
    """
    将 UDP/map 坐标系下的 (x,y) 转到“起点为原点、起点->终点为 +x”的新平面坐标系。
    """
    start, x_axis, y_axis = frame_axes_xy()
    delta = np.array([float(x_real), float(y_real)], dtype=float) - start
    x_new = float(delta @ x_axis)
    y_new = float(delta @ y_axis)
    return np.array([x_new, y_new], dtype=float)


def real_xy_to_sim_world_xy(x_real: float, y_real: float, sim_origin_xy: np.ndarray) -> np.ndarray:
    """
    将 UDP/map 坐标转换到 MuJoCo 世界坐标的平面坐标:
        sim_xy = sim_origin_xy + new_frame_xy
    """
    sim_origin_xy = np.asarray(sim_origin_xy, dtype=float).reshape(2)
    rel_xy = real_xy_to_new_frame_xy(x_real, y_real)
    return sim_origin_xy + rel_xy


def target_world_in_sim(sim_origin_xyz: np.ndarray) -> np.ndarray:
    """
    读取配置中的目标点并转换到 MuJoCo 世界坐标。
    """
    sim_origin_xyz = np.asarray(sim_origin_xyz, dtype=float).reshape(3)
    tgt = np.asarray(SIM_TARGET_POS_REAL, dtype=float).reshape(3)
    tgt_xy_world = real_xy_to_sim_world_xy(float(tgt[0]), float(tgt[1]), sim_origin_xyz[:2])
    return np.array([float(tgt_xy_world[0]), float(tgt_xy_world[1]), float(tgt[2])], dtype=float)


def target_in_robot_frame(sim_origin_xyz: np.ndarray) -> np.ndarray:
    """
    返回配置目标点在机械臂/底盘坐标系下的位置（基于仿真起点时刻）。
    """
    sim_origin_xyz = np.asarray(sim_origin_xyz, dtype=float).reshape(3)
    return target_world_in_sim(sim_origin_xyz) - sim_origin_xyz


def build_command(seq_num: int, cmd: int, payload: bytes = b"") -> bytes:
    """
    构建通用请求包：16字节授权码 + 12字节协议头 + payload
    """
    if not (0 <= seq_num <= 0xFFFF):
        raise ValueError(f"seq_num out of range: {seq_num}")
    if not (0 <= cmd <= 0xFF):
        raise ValueError(f"cmd out of range: {cmd}")
    if len(payload) > 0xFFFF:
        raise ValueError(f"payload too large: {len(payload)}")

    header = struct.pack(
        _HEADER_FMT,
        0x01,  # 版本号
        0x00,  # 报文类型：请求
        seq_num,
        _SERVICE_CODE,
        cmd,
        0x00,  # 执行码：请求填 0
        0x00,  # 保留
        len(payload),  # 数据区长度
        0x00,  # 保留
    )
    return AUTH_CODE + header + payload


def build_0x17_command(seq_num):
    """
    构建查询命令包 (0x17)
    包含 16字节授权码 + 12字节协议头
    """
    return build_command(seq_num, 0x17, payload=b"")


def build_0x15_command(seq_num: int) -> bytes:
    return build_command(seq_num, 0x15, payload=b"")


def build_0x1f_confirm_location_command(seq_num: int) -> bytes:
    return build_command(seq_num, 0x1F, payload=b"")


def build_0x11_switch_mode_command(seq_num: int, *, automatic: bool) -> bytes:
    # C++: quint8 AGVdata[4] = {1,0,0,0} 表示自动；{0,0,0,0} 表示手动
    payload = bytes([1 if automatic else 0, 0, 0, 0])
    return build_command(seq_num, 0x11, payload=payload)


def build_0x14_manual_localize_command(seq_num: int, *, x: float, y: float, theta: float) -> bytes:
    # C++ 使用 3 个 little-endian double: Tx, Ty, Ta
    payload = struct.pack("<ddd", float(x), float(y), float(theta))
    return build_command(seq_num, 0x14, payload=payload)


def build_0x16_navigation_control_command(
    seq_num: int,
    *,
    nav_mode: int = 0,
    target_point_id: str,
    path_point_ids: list[int],
) -> bytes:
    """
    0x16 导航控制（按 kcUDP.cpp 的 QDataStream LittleEndian 结构构建）。
    当前仅实现：开始导航 + 导航到路径点 + 指定路径点序列。
    """
    if nav_mode not in (0, 1, 2):
        raise ValueError(f"nav_mode must be 0/1/2, got {nav_mode}")
    if nav_mode == 0 and not target_point_id:
        raise ValueError("target_point_id must be non-empty when nav_mode==0")
    if any((pid < 0 or pid > 0xFFFF) for pid in path_point_ids):
        raise ValueError(f"path_point_ids out of range: {path_point_ids}")
    if len(path_point_ids) > 128:
        raise ValueError(f"path_point_ids too long: {len(path_point_ids)} (max 128)")

    # data1..data17 与 kcUDP.cpp 对齐
    data1 = 0  # 操作类型：0 开始导航
    data2 = int(nav_mode)  # 导航方式：0到路径点/1到路径上的点/2自由坐标点
    data3 = 1  # 是否指定导航路径：1 指定
    data4 = 0  # 是否启用交通管理：0 不启用

    data5_bytes = target_point_id.encode("ascii", errors="ignore")[:8].ljust(8, b"\x00")

    data6 = 0  # 目标路径起点ID（导航方式 1 用）
    data7 = 0  # 目标路径终点ID（导航方式 1 用）
    data8 = 0.0  # 目标点x（导航方式 1/2 用）
    data9 = 0.0  # 目标点y
    data10 = 0.0  # 目标点theta

    data11_bytes = b"\x00\x00"

    data12 = int(len(path_point_ids))
    data13 = list(path_point_ids) + [0] * (128 - len(path_point_ids))

    data14_bytes = b"\x00" * 12

    data15 = 0  # 禁止通行路径数量
    data16_bytes = b"\x00" * 3
    data17 = [0] * 64

    payload = bytearray(432)
    offset = 0

    struct.pack_into("<BBBB", payload, offset, data1, data2, data3, data4)
    offset += 4
    payload[offset : offset + 8] = data5_bytes
    offset += 8

    struct.pack_into("<HH", payload, offset, data6, data7)
    offset += 4

    struct.pack_into("<fff", payload, offset, float(data8), float(data9), float(data10))
    offset += 12

    payload[offset : offset + 2] = data11_bytes
    offset += 2

    struct.pack_into("<H", payload, offset, data12)
    offset += 2

    struct.pack_into("<" + "H" * 128, payload, offset, *data13)
    offset += 256

    payload[offset : offset + 12] = data14_bytes
    offset += 12

    struct.pack_into("<B", payload, offset, data15)
    offset += 1
    payload[offset : offset + 3] = data16_bytes
    offset += 3

    struct.pack_into("<" + "H" * 64, payload, offset, *data17)
    offset += 128

    if offset != len(payload):
        raise RuntimeError(f"0x16 payload size mismatch: offset={offset}, len={len(payload)}")

    return build_command(seq_num, 0x16, payload=bytes(payload))


def _exec_code_to_str(exec_code: int) -> str:
    mapping = {
        0x00: "Success",
        0x01: "UnknownError",
        0x02: "ServiceCodeError",
        0x03: "CommandCodeError",
        0x04: "HeaderError",
        0x05: "LengthError",
        0x06: "BeyondLimit",
    }
    return mapping.get(exec_code, f"0x{exec_code:02x}")


def parse_response_header(data: bytes):
    """
    解析通用应答头，返回 (header, payload, err)。
    header 包含：version/type/seq/service/cmd/exec_code/data_len。
    """
    if len(data) < _AUTH_SIZE + _HEADER_SIZE:
        return None, None, f"packet_too_short(len={len(data)})"

    if data[:_AUTH_SIZE] != AUTH_CODE:
        return None, None, "auth_code_mismatch"

    try:
        version, msg_type, seq, service, cmd, exec_code, _, data_len, _ = struct.unpack(
            _HEADER_FMT,
            data[_AUTH_SIZE : _AUTH_SIZE + _HEADER_SIZE],
        )
    except struct.error as e:
        return None, None, f"header_unpack_error({e})"

    if version != 0x01 or msg_type != 0x01:
        return None, None, f"unexpected_header(version=0x{version:02x}, type=0x{msg_type:02x})"

    if service != _SERVICE_CODE:
        return None, None, f"unexpected_service(0x{service:02x})"

    total_len = _AUTH_SIZE + _HEADER_SIZE + data_len
    if len(data) < total_len:
        return None, None, f"truncated_payload(need={total_len}, got={len(data)})"

    payload = data[_AUTH_SIZE + _HEADER_SIZE : total_len]
    header = {
        "version": version,
        "msg_type": msg_type,
        "seq": seq,
        "service": service,
        "cmd": cmd,
        "exec_code": exec_code,
        "data_len": data_len,
    }
    return header, payload, None


def parse_0x17_response(data: bytes, expected_seq: int):
    """
    解析 0x17 应答，返回 (parsed, err)。
    - x, y: 地图坐标系下位置（单位通常为 m）
    - theta: 位置的朝向角（通常 rad）
    - v: 前进速度（通常 m/s）
    - w: 转弯速度（通常 rad/s）
    """
    header, payload, err = parse_response_header(data)
    if header is None:
        return None, err

    seq = header["seq"]
    cmd = header["cmd"]
    exec_code = header["exec_code"]

    if seq != expected_seq or cmd != 0x17:
        return None, f"unexpected_seq_or_cmd(seq={seq}, expected={expected_seq}, cmd=0x{cmd:02x})"
    if exec_code != 0x00:
        return None, f"exec_code_error({_exec_code_to_str(exec_code)})"

    try:
        x = struct.unpack_from("<d", payload, 0x08)[0]
        y = struct.unpack_from("<d", payload, 0x10)[0]
        theta = struct.unpack_from("<d", payload, 0x18)[0]
        v = struct.unpack_from("<d", payload, 0x30)[0]
        w = struct.unpack_from("<d", payload, 0x38)[0]
    except struct.error as e:
        return None, f"payload_unpack_error({e})"

    # 以下字段用于导航初始化（偏移来自 kcUDP.cpp 的解析逻辑）
    run_mode = payload[0x2A] if len(payload) > 0x2A else None  # 0手动/1自动
    map_load_status = payload[0x2B] if len(payload) > 0x2B else None  # 0成功/1失败/2未载入/3载入中
    task_status = payload[0x50] if len(payload) > 0x50 else None  # 0无任务/1等待/2前往/...
    positioning_status = payload[0x70] if len(payload) > 0x70 else None  # 0失败/1成功/2定位中/3定位成功(按C++注释)
    confidence = None
    if len(payload) >= 0xA4 + 8:
        try:
            confidence = struct.unpack_from("<d", payload, 0xA4)[0]
        except struct.error:
            confidence = None

    return (
        {
            "seq": seq,
            "x": x,
            "y": y,
            "theta": theta,
            "v": v,
            "w": w,
            "run_mode": run_mode,
            "map_load_status": map_load_status,
            "task_status": task_status,
            "positioning_status": positioning_status,
            "confidence": confidence,
        },
        None,
    )


def parse_0x15_response(data: bytes, expected_seq: int):
    """
    解析 0x15 应答，返回 (parsed, err)。
    - x, y: 地图坐标系下位置（单位通常为 m）
    - angle: 朝向角（通常 rad）
    - confidence: 置信度（通常 0~1）
    """
    header, payload, err = parse_response_header(data)
    if header is None:
        return None, err

    seq = header["seq"]
    cmd = header["cmd"]
    exec_code = header["exec_code"]

    if seq != expected_seq or cmd != 0x15:
        return None, f"unexpected_seq_or_cmd(seq={seq}, expected={expected_seq}, cmd=0x{cmd:02x})"
    if exec_code != 0x00:
        return None, f"exec_code_error({_exec_code_to_str(exec_code)})"

    try:
        x = struct.unpack_from("<d", payload, 0x00)[0]
        y = struct.unpack_from("<d", payload, 0x08)[0]
        angle = struct.unpack_from("<d", payload, 0x10)[0]
        confidence = struct.unpack_from("<d", payload, 0x18)[0]
    except struct.error as e:
        return None, f"payload_unpack_error({e})"

    return {"seq": seq, "x": x, "y": y, "angle": angle, "confidence": confidence}, None


def _run_mode_str(v: Optional[int]) -> str:
    if v is None:
        return "NA"
    return {0: "MANUAL", 1: "AUTO"}.get(v, f"UNK({v})")


def _map_load_str(v: Optional[int]) -> str:
    if v is None:
        return "NA"
    return {0: "OK", 1: "FAIL", 2: "NOT_LOADED", 3: "LOADING"}.get(v, f"UNK({v})")


def _task_status_str(v: Optional[int]) -> str:
    if v is None:
        return "NA"
    return {
        0: "NO_TASK",
        1: "WAITING",
        2: "MOVING",
        3: "PAUSED",
        4: "DONE",
        5: "FAILED",
        6: "EXIT",
        7: "WAIT_DOOR",
    }.get(v, f"UNK({v})")


def _positioning_str(v: Optional[int]) -> str:
    if v is None:
        return "NA"
    return {0: "FAIL", 1: "OK", 2: "LOCATING", 3: "OK2"}.get(v, f"UNK({v})")


def _prompt_enter(prompt: str) -> bool:
    """
    交互确认：
      - 直接按 Enter：继续
      - 输入 q/quit/exit 再回车：退出
    非交互（stdin 不可用）时默认继续。
    """
    try:
        ans = input(prompt)
    except EOFError:
        return True
    if ans is None:
        return True
    ans = ans.strip().lower()
    return ans not in ("q", "quit", "exit")


def _prompt_choice(prompt: str, *, default: str) -> str:
    """
    返回用户输入的选择（小写），空输入返回 default。
    非交互（stdin 不可用）时返回 default。
    """
    try:
        ans = input(prompt)
    except EOFError:
        return default
    if ans is None:
        return default
    ans = ans.strip().lower()
    return default if ans == "" else ans


def _recv_matching(
    sock: socket.socket,
    *,
    expected_seq: int,
    expected_cmd: int,
    timeout_s: float,
):
    prev_timeout = sock.gettimeout()
    try:
        deadline = time.monotonic() + float(timeout_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, "TIMEOUT"
            sock.settimeout(remaining)
            try:
                data, _ = sock.recvfrom(_MAX_RECV_BYTES)
            except socket.timeout:
                return None, "TIMEOUT"

            header, _, err = parse_response_header(data)
            if header is None:
                return None, err
            if header["seq"] != expected_seq or header["cmd"] != expected_cmd:
                continue
            return data, None
    finally:
        sock.settimeout(prev_timeout)


def request_0x17(sock: socket.socket, seq: int, *, timeout_s: float):
    packet = build_0x17_command(seq)
    sock.sendto(packet, (ROBOT_IP, ROBOT_PORT))
    data, err = _recv_matching(sock, expected_seq=seq, expected_cmd=0x17, timeout_s=timeout_s)
    if data is None:
        return None, err
    return parse_0x17_response(data, expected_seq=seq)


def request_0x15(sock: socket.socket, seq: int, *, timeout_s: float):
    packet = build_0x15_command(seq)
    sock.sendto(packet, (ROBOT_IP, ROBOT_PORT))
    data, err = _recv_matching(sock, expected_seq=seq, expected_cmd=0x15, timeout_s=timeout_s)
    if data is None:
        return None, err
    return parse_0x15_response(data, expected_seq=seq)


def request_ack(sock: socket.socket, seq: int, *, packet: bytes, expected_cmd: int, timeout_s: float):
    sock.sendto(packet, (ROBOT_IP, ROBOT_PORT))
    data, err = _recv_matching(sock, expected_seq=seq, expected_cmd=expected_cmd, timeout_s=timeout_s)
    if data is None:
        return False, err

    header, _, err = parse_response_header(data)
    if header is None:
        return False, err
    if header["exec_code"] != 0x00:
        return False, f"exec_code_error({_exec_code_to_str(header['exec_code'])})"
    return True, None


def init_navigation_then_stream(
    sock: socket.socket,
    *,
    sequence: int,
    confidence_threshold: float,
    manual_localize_pose: Optional[tuple[float, float, float]],
    nav_target_point_id: str,
    nav_path_point_ids: list[int],
    nav_mode: int,
    interactive: bool,
):
    """
    参考 kcUDP.cpp 的控制逻辑：
      1) 轮询 0x17，等待地图载入成功 + 定位成功 + 置信度足够
      2) 0x1F 确认位置
      3) 0x11 切换自动模式
      4) 0x16 下发路径点导航

    返回更新后的 sequence。
    """
    print("\n=== 导航初始化（分步确认） ===")
    if interactive:
        print("说明：每个步骤开始前都会提示按 Enter；输入 q 回车可随时取消。")
    print(f"- confidence_threshold={confidence_threshold:.2f}")
    print(f"- nav_target_point_id={nav_target_point_id!r}")
    print(f"- nav_path_point_ids={nav_path_point_ids}")
    print(f"- nav_mode={nav_mode}")
    if manual_localize_pose is None:
        print("- manual_localize_pose=DISABLED")
    else:
        mx, my, ma = manual_localize_pose
        print(f"- manual_localize_pose=({mx:.3f},{my:.3f},{ma:.3f})")
    print("")

    last_status_print_t = 0.0

    # Step 1: 等地图载入成功
    if interactive and not _prompt_enter("[STEP 1/5] 等待地图载入成功（轮询 0x17）。按 Enter 开始（q 退出）: "):
        print("已取消导航初始化。")
        return None

    print("[STEP 1/5] 开始：等待地图载入成功（map_load_status==0）...")
    while True:
        parsed, err = request_0x17(sock, sequence, timeout_s=SOCKET_TIMEOUT_S)
        sequence = (sequence + 1) % 65536

        now = time.monotonic()
        if parsed is None:
            if now - last_status_print_t > 1.0:
                print(f"  - 0x17 TIMEOUT/PARSE_FAIL: {err}")
                last_status_print_t = now
            time.sleep(INIT_QUERY_PERIOD_S)
            continue

        map_load = parsed.get("map_load_status")
        if now - last_status_print_t > 1.0:
            print(
                f"  - map={_map_load_str(map_load)} | run={_run_mode_str(parsed.get('run_mode'))} | "
                f"pos={_positioning_str(parsed.get('positioning_status'))} | "
                f"conf={parsed.get('confidence') if parsed.get('confidence') is not None else 'NA'} | "
                f"task={_task_status_str(parsed.get('task_status'))}"
            )
            last_status_print_t = now

        if map_load == 0:
            print("[STEP 1/5] 完成：地图载入成功")
            break

        time.sleep(INIT_QUERY_PERIOD_S)

    # Step 2: 等定位成功 + 置信度足够（优先使用 0x15 的 confidence；0x17 的 confidence 可能恒为 0）
    if interactive and not _prompt_enter(
        f"[STEP 2/5] 下一步：确认定位成功 + 置信度>= {confidence_threshold:.2f}（轮询 0x17/0x15）。按 Enter 继续（q 退出）: "
    ):
        print("已取消导航初始化。")
        return None

    print(f"[STEP 2/5] 开始：等待定位成功 + 置信度>= {confidence_threshold:.2f} ...")
    last_status_print_t = 0.0
    last_manual_prompt_t = 0.0
    last_conf_15 = None
    last_loc_15 = None
    while True:
        status_17, err_17 = request_0x17(sock, sequence, timeout_s=SOCKET_TIMEOUT_S)
        sequence = (sequence + 1) % 65536

        now = time.monotonic()
        if status_17 is None:
            if now - last_status_print_t > 1.0:
                print(f"  - 0x17 TIMEOUT/PARSE_FAIL: {err_17}")
                last_status_print_t = now
            time.sleep(INIT_QUERY_PERIOD_S)
            continue

        positioning_status = status_17.get("positioning_status")
        map_load = status_17.get("map_load_status")

        localized_ok = positioning_status in (1, 3)

        # 实车上定位置信度通常来自 0x15（kcUDP.cpp: confidenceUDP），0x17 的 confidence 可能恒为 0。
        confidence_17 = status_17.get("confidence")
        confidence_15 = None
        if map_load == 0:
            status_15, err_15 = request_0x15(sock, sequence, timeout_s=SOCKET_TIMEOUT_S)
            sequence = (sequence + 1) % 65536
            if status_15 is not None:
                confidence_15 = status_15.get("confidence")
                last_conf_15 = confidence_15
                last_loc_15 = (status_15.get("x"), status_15.get("y"))
            else:
                # 保留上一次 0x15 成功的值，避免偶发超时导致一直显示 NA
                confidence_15 = last_conf_15

        confidence = confidence_15 if confidence_15 is not None else confidence_17
        conf_ok = (confidence is not None) and (confidence >= confidence_threshold)

        if now - last_status_print_t > 1.0:
            loc_msg = f"({status_17['x']:.3f},{status_17['y']:.3f})"
            loc15_msg = "NA"
            if last_loc_15 is not None and last_loc_15[0] is not None and last_loc_15[1] is not None:
                loc15_msg = f"({last_loc_15[0]:.3f},{last_loc_15[1]:.3f})"
            print(
                f"  - map={_map_load_str(map_load)} | pos={_positioning_str(positioning_status)} | "
                f"conf={confidence if confidence is not None else 'NA'} "
                f"(0x15={confidence_15 if confidence_15 is not None else 'NA'}, 0x17={confidence_17 if confidence_17 is not None else 'NA'}) | "
                f"loc17={loc_msg} | loc15={loc15_msg}"
            )
            last_status_print_t = now

        if map_load != 0:
            time.sleep(INIT_QUERY_PERIOD_S)
            continue

        if localized_ok and conf_ok:
            print("[STEP 2/5] 完成：定位成功且置信度满足阈值")
            break

        # 可选：定位/置信度不足时，允许发送 0x14 手动定位（需要用户确认）
        if manual_localize_pose is not None and positioning_status in (0, 1, 3):
            now2 = time.monotonic()
            if (not interactive) or (now2 - last_manual_prompt_t > 2.0):
                mx, my, ma = manual_localize_pose
                if interactive:
                    choice = _prompt_choice(
                        f"  - 当前未达标，可选发送 0x14 手动定位 ({mx:.3f},{my:.3f},{ma:.3f})。"
                        " [s=发送 / Enter=继续等待 / q=退出]: ",
                        default="",
                    )
                    if choice in ("q", "quit", "exit"):
                        print("已取消导航初始化。")
                        return None
                    do_send = choice in ("s", "send", "y", "yes")
                else:
                    do_send = True

                last_manual_prompt_t = now2
                if do_send:
                    print("  - 发送 0x14 手动定位 ...")
                    ok, ack_err = request_ack(
                        sock,
                        sequence,
                        packet=build_0x14_manual_localize_command(sequence, x=mx, y=my, theta=ma),
                        expected_cmd=0x14,
                        timeout_s=SOCKET_TIMEOUT_S,
                    )
                    sequence = (sequence + 1) % 65536
                    if ok:
                        print("  - 0x14 执行成功，继续等待定位收敛...")
                    else:
                        print(f"  - 0x14 失败：{ack_err}")

        time.sleep(INIT_QUERY_PERIOD_S)

    # Step 3: 确认位置（0x1F）——通常在手动模式下执行
    while True:
        status_17, err_17 = request_0x17(sock, sequence, timeout_s=SOCKET_TIMEOUT_S)
        sequence = (sequence + 1) % 65536
        if status_17 is None:
            print(f"[STEP 3/5] 0x17 FAIL: {err_17}")
            time.sleep(INIT_QUERY_PERIOD_S)
            continue
        run_mode = status_17.get("run_mode")
        if run_mode is None:
            print("[STEP 3/5] run_mode=NA，继续等待 ...")
            time.sleep(INIT_QUERY_PERIOD_S)
            continue
        break

    if run_mode == 0:
        while True:
            if interactive and not _prompt_enter("[STEP 3/5] 将发送 0x1F 确认位置。按 Enter 发送（q 退出）: "):
                print("已取消导航初始化。")
                return None
            print("[STEP 3/5] 发送 0x1F 确认位置 ...")
            ok, ack_err = request_ack(
                sock,
                sequence,
                packet=build_0x1f_confirm_location_command(sequence),
                expected_cmd=0x1F,
                timeout_s=SOCKET_TIMEOUT_S,
            )
            sequence = (sequence + 1) % 65536
            if ok:
                print("[STEP 3/5] 完成：0x1F 执行成功")
                break
            print(f"[STEP 3/5] 失败：{ack_err}")
            if interactive:
                choice = _prompt_choice("  - Enter 重试 / q 退出: ", default="retry")
                if choice in ("q", "quit", "exit"):
                    print("已取消导航初始化。")
                    return None
            else:
                raise RuntimeError(f"0x1F 确认位置失败：{ack_err}")
    else:
        print("[STEP 3/5] 跳过：当前已是 AUTO 模式（不发送 0x1F）")

    # Step 4: 切换自动模式（0x11）
    while True:
        status_17, err_17 = request_0x17(sock, sequence, timeout_s=SOCKET_TIMEOUT_S)
        sequence = (sequence + 1) % 65536
        if status_17 is None:
            print(f"[STEP 4/5] 0x17 FAIL: {err_17}")
            time.sleep(INIT_QUERY_PERIOD_S)
            continue
        run_mode = status_17.get("run_mode")
        if run_mode is None:
            print("[STEP 4/5] run_mode=NA，继续等待 ...")
            time.sleep(INIT_QUERY_PERIOD_S)
            continue
        break

    if run_mode != 1:
        while True:
            if interactive and not _prompt_enter("[STEP 4/5] 将发送 0x11 切换自动模式。按 Enter 发送（q 退出）: "):
                print("已取消导航初始化。")
                return None
            print("[STEP 4/5] 发送 0x11 切换自动模式 ...")
            ok, ack_err = request_ack(
                sock,
                sequence,
                packet=build_0x11_switch_mode_command(sequence, automatic=True),
                expected_cmd=0x11,
                timeout_s=SOCKET_TIMEOUT_S,
            )
            sequence = (sequence + 1) % 65536
            if ok:
                break
            print(f"[STEP 4/5] 失败：{ack_err}")
            if interactive:
                choice = _prompt_choice("  - Enter 重试 / q 退出: ", default="retry")
                if choice in ("q", "quit", "exit"):
                    print("已取消导航初始化。")
                    return None
            else:
                raise RuntimeError(f"0x11 切换自动失败：{ack_err}")

        print("[STEP 4/5] 等待 run_mode==AUTO ...")
        while True:
            status_17, err_17 = request_0x17(sock, sequence, timeout_s=SOCKET_TIMEOUT_S)
            sequence = (sequence + 1) % 65536
            if status_17 is None:
                print(f"  - 0x17 FAIL: {err_17}")
                time.sleep(INIT_QUERY_PERIOD_S)
                continue
            run_mode = status_17.get("run_mode")
            if run_mode == 1:
                print("[STEP 4/5] 完成：已进入 AUTO 模式")
                break
            time.sleep(INIT_QUERY_PERIOD_S)
    else:
        print("[STEP 4/5] 跳过：已是 AUTO 模式（不发送 0x11）")

    # Step 5: 最后确认 task_status；若无任务则发送 0x16，并开始 0x17 状态循环
    while True:
        status_17, err_17 = request_0x17(sock, sequence, timeout_s=SOCKET_TIMEOUT_S)
        sequence = (sequence + 1) % 65536
        if status_17 is None:
            print(f"[STEP 5/5] 0x17 FAIL: {err_17}")
            time.sleep(INIT_QUERY_PERIOD_S)
            continue
        task_status = status_17.get("task_status")
        if task_status is None:
            print("[STEP 5/5] task_status=NA，继续等待 ...")
            time.sleep(INIT_QUERY_PERIOD_S)
            continue
        break

    if task_status == 0:
        while True:
            if interactive and not _prompt_enter(
                f"[STEP 5/5] 当前无任务，将发送 0x16（nav_mode={nav_mode}, path={nav_path_point_ids}）并启动状态流。"
                " 按 Enter 发送（q 退出）: "
            ):
                print("已取消导航初始化。")
                return None

            packet_16 = build_0x16_navigation_control_command(
                sequence,
                nav_mode=nav_mode,
                target_point_id=nav_target_point_id,
                path_point_ids=nav_path_point_ids,
            )
            print("[STEP 5/5] 发送 0x16 开始路径点导航 ...")
            ok, ack_err = request_ack(
                sock,
                sequence,
                packet=packet_16,
                expected_cmd=0x16,
                timeout_s=SOCKET_TIMEOUT_S,
            )
            sequence = (sequence + 1) % 65536
            if ok:
                print("[STEP 5/5] 完成：0x16 已执行，即将开始同步获取 0x17 状态\n")
                break
            print(f"[STEP 5/5] 失败：{ack_err}")
            if interactive:
                choice = _prompt_choice("  - Enter 重试 / q 退出: ", default="retry")
                if choice in ("q", "quit", "exit"):
                    print("已取消导航初始化。")
                    return None
            else:
                raise RuntimeError(f"0x16 下发导航失败：{ack_err}")
    else:
        if interactive and not _prompt_enter(
            f"[STEP 5/5] 检测到已有任务 task={_task_status_str(task_status)}，不会发送 0x16。按 Enter 开始状态流（q 退出）: "
        ):
            print("已取消导航初始化。")
            return None
        print("[STEP 5/5] 完成：跳过 0x16（已有任务），即将开始同步获取 0x17 状态\n")

    return sequence

def main():
    global ROBOT_IP, ROBOT_PORT, POLL_PERIOD_S, AUTH_CODE

    import argparse

    parser = argparse.ArgumentParser(description="底盘 UDP 通信：状态获取 + (可选)导航初始化")
    parser.add_argument("--ip", default=ROBOT_IP)
    parser.add_argument("--port", type=int, default=ROBOT_PORT)
    parser.add_argument("--poll", type=float, default=POLL_PERIOD_S, help="0x17 状态轮询周期（秒）")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="自动连续运行：不需要按 Enter 分步确认/不需要逐步轮询（默认是按 Enter 一步步执行）",
    )
    parser.add_argument(
        "--auth",
        default=None,
        help="16字节授权码 HEX（支持空格/短横线），例如: '31 04 ...' 或 '31-04-...' 或 '3104...'",
    )
    nav_group = parser.add_mutually_exclusive_group()
    nav_group.add_argument(
        "--init-nav",
        dest="init_nav",
        action="store_true",
        help="在开始持续获取 0x17 之前，按 kcUDP.cpp 的流程做导航初始化并下发 0x16 路径点任务",
    )
    nav_group.add_argument(
        "--no-init-nav",
        dest="init_nav",
        action="store_false",
        help="不做导航初始化，仅轮询获取 0x17 状态",
    )
    parser.set_defaults(init_nav=True)
    parser.add_argument("--confidence-threshold", type=float, default=INIT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--nav-mode", type=int, default=DEFAULT_NAV_MODE, choices=(0, 1, 2))
    parser.add_argument("--nav-target-id", default=DEFAULT_NAV_TARGET_POINT_ID)
    parser.add_argument(
        "--nav-path-ids",
        default=",".join(str(x) for x in DEFAULT_NAV_PATH_POINT_IDS),
        help="逗号分隔的路径点 ID（最多 128 个），例如: 1,2,3,4,5",
    )
    parser.add_argument(
        "--manual-localize",
        default=None,
        help="可选：触发 0x14 手动定位 (x,y,theta)，例如: 0.05,0.038,0.0",
    )
    parser.add_argument(
        "--no-user-confirm",
        action="store_true",
        help="执行 --init-nav 时不需要按 Enter 分步确认（自动执行）",
    )
    args = parser.parse_args()

    if args.auth is not None:
        cleaned = str(args.auth).replace(" ", "").replace("-", "").strip()
        try:
            raw = bytes.fromhex(cleaned)
        except ValueError as e:
            raise ValueError(f"--auth 不是合法的 HEX：{e}") from e
        if len(raw) != 16:
            raise ValueError(f"--auth 必须是 16 字节，当前是 {len(raw)} 字节")
        AUTH_CODE = raw

    ROBOT_IP = str(args.ip)
    ROBOT_PORT = int(args.port)
    POLL_PERIOD_S = float(args.poll)

    nav_path_ids: list[int] = []
    if str(args.nav_path_ids).strip():
        nav_path_ids = [int(s) for s in str(args.nav_path_ids).split(",") if s.strip()]
    if not nav_path_ids:
        nav_path_ids = DEFAULT_NAV_PATH_POINT_IDS.copy()

    manual_localize_pose: Optional[tuple[float, float, float]] = None
    if args.manual_localize is not None:
        parts = [p.strip() for p in str(args.manual_localize).split(",")]
        if len(parts) != 3:
            raise ValueError("--manual-localize 需要 3 个值: x,y,theta")
        manual_localize_pose = (float(parts[0]), float(parts[1]), float(parts[2]))

    # 1. 创建 UDP Socket (不需要 Connect，直接创建)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(SOCKET_TIMEOUT_S)  # 防止收不到数据卡死
    
    print(f"正在尝试连接机器人 {ROBOT_IP}:{ROBOT_PORT} ...")
    if args.auto:
        print("运行模式：AUTO（连续运行，不需要按 Enter）")
        print("按 Ctrl+C 停止程序\n")
    else:
        print("运行模式：STEP（默认按 Enter 一步步执行；输入 q 回车可退出）\n")

    t_start = time.monotonic()
    sequence = 0
    last_ok_t = None

    sent_count = 0
    ok_count = 0
    timeout_count = 0
    parse_fail_count = 0

    try:
        if args.init_nav:
            new_sequence = init_navigation_then_stream(
                client,
                sequence=sequence,
                confidence_threshold=float(args.confidence_threshold),
                manual_localize_pose=manual_localize_pose,
                nav_target_point_id=str(args.nav_target_id),
                nav_path_point_ids=nav_path_ids,
                nav_mode=int(args.nav_mode),
                interactive=(not args.auto) and (not args.no_user_confirm),
            )
            if new_sequence is None:
                return
            sequence = new_sequence
            # 连续输出阶段从“第一个成功应答”开始统计 dt
            last_ok_t = None

        while True:
            if not args.auto:
                if not _prompt_enter("[STREAM] 按 Enter 获取下一帧状态（q 退出）: "):
                    break

            # --- 发送环节 ---
            # 构建报文
            packet = build_0x17_command(sequence)
            # 直接发送给目标 IP
            client.sendto(packet, (ROBOT_IP, ROBOT_PORT))
            sent_count += 1
            
            # --- 接收环节 ---
            try:
                # 阻塞等待回复
                data, addr = client.recvfrom(2048)
                
                now = time.monotonic()
                t_rel = now - t_start

                parsed, err = parse_0x17_response(data, expected_seq=sequence)
                if parsed is None:
                    parse_fail_count += 1
                    print(
                        f"[Seq:{sequence}] t={t_rel:.3f}s | PARSE_FAIL: {err} | "
                        f"len={len(data)} | sent={sent_count} ok={ok_count} timeout={timeout_count} fail={parse_fail_count}"
                    )
                    continue

                ok_count += 1
                dt_ok = None if last_ok_t is None else (now - last_ok_t)
                last_ok_t = now

                x = parsed["x"]
                y = parsed["y"]
                theta = parsed["theta"]
                v = parsed["v"]
                w = parsed["w"]

                dt_s = "NA" if dt_ok is None else f"{dt_ok:.3f}s"
                print(
                    f"[Seq:{sequence}] t={t_rel:.3f}s dt={dt_s} | "
                    f"(x,y)=({x:.3f},{y:.3f}) m | theta={theta:.3f} rad | v={v:.3f} m/s | w={w:.3f} rad/s"
                )
                
            except socket.timeout:
                timeout_count += 1
                now = time.monotonic()
                t_rel = now - t_start
                print(
                    f"[Seq:{sequence}] t={t_rel:.3f}s | TIMEOUT(after={SOCKET_TIMEOUT_S:.1f}s) | "
                    f"sent={sent_count} ok={ok_count} timeout={timeout_count} fail={parse_fail_count}"
                )

            # 序列号递增 (模拟真实通信)
            sequence = (sequence + 1) % 65536
            
            # 控制频率：按 POLL_PERIOD_S 轮询（STEP 模式由用户按键节流，这里不 sleep）
            if args.auto:
                time.sleep(POLL_PERIOD_S)

    except KeyboardInterrupt:
        print("\n程序已停止。")
    finally:
        client.close()

if __name__ == "__main__":
    main()
