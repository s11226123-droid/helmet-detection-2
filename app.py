from collections import Counter, deque
import os
import tempfile
import cv2
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ============================================================
# 1. 頁面標題與基本配置
# ============================================================
st.set_page_config(page_title="機車未戴安全帽檢測系統", layout="wide")
st.title("🛵 三階段機車騎士未戴安全帽自動辨識與自動違規截圖系統")


@st.cache_resource
def load_models(m1_path, m2_path, m3_path):
    m1 = YOLO(m1_path)
    m2 = YOLO(m2_path)
    m3 = YOLO(m3_path)
    return m1, m2, m3


# ============================================================
# 2. 側邊欄：動態調整參數設定
# ============================================================
st.sidebar.header("⚙️ 系統參數設定")

# 模型路徑
m1_path = st.sidebar.text_input(
    "Stage 1 模型路徑", "models/yolov11s_best.pt"
)
m2_path = st.sidebar.text_input("Stage 2 模型路徑", "models/best0831.pt")
m3_path = st.sidebar.text_input(
    "Stage 3 模型路徑", "models/bestcla0809.pt"
)

# 信心門檻與 Fallback 比例
conf_s1 = st.sidebar.slider(
    "Stage 1 騎士辨識門檻", 0.1, 1.0, 0.9, 0.05
)
conf_s2 = st.sidebar.slider(
    "Stage 2 頭部辨識門檻", 0.1, 1.0, 0.7, 0.05
)
conf_s3 = st.sidebar.slider(
    "Stage 3 安全帽分類門檻", 0.1, 1.0, 0.9, 0.05
)
fallback_ratio = st.sidebar.slider("Fallback 切割比例", 0.1, 0.5, 0.35, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("🔒 狀態鎖定與違規截圖設定")

# 新增：狀態鎖定門檻
lock_threshold = st.sidebar.number_input(
    "連續辨識幾影格後鎖定騎士狀態 (Lock Status)",
    min_value=1,
    max_value=30,
    value=5,
    step=1,
    help="當同一個 ID 連續被辨識為 helmet 或 without helmet 達到此幀數時，將鎖定其狀態不再改變。",
)

# 連續未戴安全帽截圖門檻
screenshot_threshold = st.sidebar.number_input(
    "連續未戴安全帽幾影格觸發截圖",
    min_value=1,
    max_value=30,
    value=5,
    step=1,
)

# 檢查模型檔案是否存在並載入
if (
    os.path.exists(m1_path)
    and os.path.exists(m2_path)
    and os.path.exists(m3_path)
):
    model_stage1, model_stage2, model_stage3 = load_models(
        m1_path, m2_path, m3_path
    )
    st.sidebar.success("✅ 三個模型成功載入")
else:
    st.sidebar.error("❌ 找不到模型檔案，請確認 models/ 資料夾內是否有 .pt 檔")
    st.stop()


# ============================================================
# 3. 核心 logic 輔助函式 (包含連續偵測鎖定邏輯)
# ============================================================
def predict_and_update_status(
    crop_img,
    track_id,
    model_s3,
    conf_s3_val,
    track_history,
    track_locked,
    lock_thresh=5,
):
    """
    進行 Stage 3 分類，並根據連續辨識結果鎖定 track_id 的狀態
    """
    # 1. 若該 ID 已經被鎖定狀態，直接回傳鎖定結果，不再重複判定
    if track_id != -1 and track_id in track_locked:
        return track_locked[track_id], 1.0

    # 2. 模型推論 Stage 3
    results_s3 = model_s3(crop_img, conf=conf_s3_val, verbose=False)[0]
    current_raw_label = None
    final_conf_s3 = 0.0

    if results_s3.probs is not None:
        cls_id_s3 = int(results_s3.probs.top1)
        final_conf_s3 = float(results_s3.probs.top1conf)
        raw_label_s3 = model_s3.names[cls_id_s3].lower().strip()

        if final_conf_s3 >= conf_s3_val:
            if (
                "helmet" in raw_label_s3
                and "no" not in raw_label_s3
                and "without" not in raw_label_s3
            ):
                current_raw_label = "helmet"
            else:
                current_raw_label = "without helmet"

    # 若 Stage 3 信心度不足，預設為未戴安全帽 (或維持暫態)
    if current_raw_label is None:
        current_raw_label = "without helmet"

    # 3. 處理 Track ID 歷史紀錄與狀態鎖定 logic
    if track_id != -1:
        if track_id not in track_history:
            track_history[track_id] = deque(maxlen=lock_thresh)

        track_history[track_id].append(current_raw_label)

        # 檢查歷史紀錄：如果 deque 滿了，且裡面的狀態全部一致 -> 觸發鎖定 (Locking)
        if len(track_history[track_id]) == lock_thresh:
            first_status = track_history[track_id][0]
            # 確認歷史佇列中每一個預測結果都相同
            if all(status == first_status for status in track_history[track_id]):
                track_locked[track_id] = first_status
                return first_status, final_conf_s3

        # 未達到鎖定條件時，採多數決或當前最新狀態
        vote_counts = Counter(track_history[track_id])
        final_status = vote_counts.most_common(1)[0][0]
    else:
        final_status = current_raw_label

    return final_status, final_conf_s3


def draw_labeled_box(
    img, bbox, label_text, color, font_scale=0.5, thickness=1
):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w_text, h_text), _ = cv2.getTextSize(
        label_text, font, font_scale, thickness
    )
    text_y = y1 - 5 if (y1 - 5 > h_text) else y1 + h_text + 5

    cv2.rectangle(
        img,
        (x1, text_y - h_text - 2),
        (x1 + w_text, text_y + 2),
        color,
        -1,
    )
    cv2.putText(
        img,
        label_text,
        (x1, text_y),
        font,
        font_scale,
        (0, 0, 0) if color == (180, 180, 180) else (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# 4. 影片上傳區塊與辨識流程
# ============================================================
uploaded_file = st.file_uploader(
    "請選擇上傳要分析的測試影片 (支持 MP4, AVI, MOV 格式)",
    type=["mp4", "avi", "mov"],
)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.info(
        f"🎥 影片基本資訊 — 解析度: {width}x{height} | FPS: {fps:.1f} | 總影格數:"
        f" {total_frames}"
    )

    if st.button("🚀 開始執行辨識"):
        # 版面配置
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📹 非即時辨識畫面")
            st_frame = st.empty()
            progress_bar = st.progress(0)
        with col2:
            st.subheader("🚨 違規截圖紀錄")
            violations_container = st.container()

        # 追蹤與狀態紀錄器
        track_history = {}       # 紀錄每個 track_id 最近 N 幀的辨識結果
        track_locked = {}        # 紀錄已鎖定狀態的 ID -> status ("helmet" / "without helmet")
        fallback_counters = {}   # 統計連續沒偵測到頭部次數，用來切換 Fallback
        fallback_locked = set()  # 記錄強制進入 Fallback 模式的 ID

        # 📸 連續辨識與截圖控制紀錄器
        consecutive_no_helmet = {}  # 紀錄每個 track_id 連續沒戴安全帽的次數
        captured_tracks = set()     # 紀錄已經完成截圖的 track_id，避免重複截圖
        screenshots_list = []       # 儲存截圖結果供最後總覽

        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame += 1

            # Stage 1: 騎士與機車辨識與 Tracking
            results_s1 = model_stage1.track(
                frame,
                conf=conf_s1,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]

            if results_s1.boxes is not None and len(results_s1.boxes) > 0:
                for box_s1 in results_s1.boxes:
                    x1_s1, y1_s1, x2_s1, y2_s1 = map(
                        int, box_s1.xyxy[0].tolist()
                    )
                    conf_s1_val = float(box_s1.conf[0])
                    cls_id_s1 = int(box_s1.cls[0])
                    label_name_s1 = model_stage1.names[cls_id_s1]
                    s1_track_id = (
                        int(box_s1.id[0]) if box_s1.id is not None else -1
                    )

                    x1_s1, y1_s1 = max(0, x1_s1), max(0, y1_s1)
                    x2_s1, y2_s1 = min(width, x2_s1), min(height, y2_s1)
                    w_s1, h_s1 = x2_s1 - x1_s1, y2_s1 - y1_s1

                    if w_s1 <= 0 or h_s1 <= 0:
                        continue

                    crop_s1 = frame[y1_s1:y2_s1, x1_s1:x2_s1]
                    if crop_s1.size == 0:
                        continue

                    # 繪製 Stage 1 騎士外框
                    draw_labeled_box(
                        frame,
                        (x1_s1, y1_s1, x2_s1, y2_s1),
                        (
                            f"#{s1_track_id} {label_name_s1}"
                            f" {conf_s1_val:.2f}"
                            if s1_track_id != -1
                            else f"{label_name_s1} {conf_s1_val:.2f}"
                        ),
                        (180, 180, 180),
                    )

                    # 判斷是否需要直接執行 Fallback
                    is_fallback = (s1_track_id != -1) and (
                        s1_track_id in fallback_locked
                    )
                    results_s2 = None

                    if not is_fallback:
                        results_s2 = model_stage2(
                            crop_s1, conf=conf_s2, verbose=False
                        )[0]

                    status = "without helmet"
                    conf_s3_val = 0.0

                    # --------------------------------------------------------
                    # 情況 A：Stage 2 成功偵測到頭部 (Head Detected)
                    # --------------------------------------------------------
                    if (
                        results_s2 is not None
                        and results_s2.boxes is not None
                        and len(results_s2.boxes) > 0
                    ):
                        if s1_track_id != -1:
                            fallback_counters[s1_track_id] = 0

                        for box_s2 in results_s2.boxes:
                            lx1, ly1, lx2, ly2 = map(
                                int, box_s2.xyxy[0].tolist()
                            )
                            crop_s2 = crop_s1[ly1:ly2, lx1:lx2]
                            if crop_s2.size == 0:
                                continue

                            gx1, gy1 = x1_s1 + lx1, y1_s1 + ly1
                            gx2, gy2 = x1_s1 + lx2, y1_s1 + ly2

                            # 更新狀態與鎖定判定
                            status, conf_s3_val = predict_and_update_status(
                                crop_s2,
                                s1_track_id,
                                model_stage3,
                                conf_s3,
                                track_history,
                                track_locked,
                                lock_thresh=lock_threshold,
                            )
                            
                            # 標示狀態：若已鎖定則於標籤加上 [LOCKED] 提示
                            is_locked_str = "[LOCKED] " if (s1_track_id in track_locked) else ""
                            color = (
                                (0, 255, 0)
                                if status == "helmet"
                                else (0, 0, 255)
                            )
                            draw_labeled_box(
                                frame,
                                (gx1, gy1, gx2, gy2),
                                f"{is_locked_str}{status} {conf_s3_val:.2f}",
                                color,
                            )

                    # --------------------------------------------------------
                    # 情況 B：Stage 2 沒偵測到頭部，觸發 Fallback
                    # --------------------------------------------------------
                    else:
                        if s1_track_id != -1 and not is_fallback:
                            fallback_counters[s1_track_id] = (
                                fallback_counters.get(s1_track_id, 0) + 1
                            )
                            if fallback_counters[s1_track_id] >= 5:
                                fallback_locked.add(s1_track_id)

                        fallback_h = max(
                            1, min(h_s1, int(h_s1 * fallback_ratio))
                        )
                        crop_fb = crop_s1[0:fallback_h, :]

                        if crop_fb.size != 0:
                            # 更新狀態與鎖定判定 (Fallback 情況下同樣套用鎖定機制)
                            status, conf_s3_val = predict_and_update_status(
                                crop_fb,
                                s1_track_id,
                                model_stage3,
                                conf_s3,
                                track_history,
                                track_locked,
                                lock_thresh=lock_threshold,
                            )
                            
                            is_locked_str = "[LOCKED] " if (s1_track_id in track_locked) else ""
                            color = (
                                (0, 255, 0)
                                if status == "helmet"
                                else (0, 0, 255)
                            )
                            draw_labeled_box(
                                frame,
                                (x1_s1, y1_s1, x2_s1, y1_s1 + fallback_h),
                                f"{is_locked_str}[FB] {status} {conf_s3_val:.2f}",
                                color,
                            )

                    # ------------------------------------------------------------
                    # 📸 核心 logic：違規連續次數統計與自動截圖
                    # ------------------------------------------------------------
                    if s1_track_id != -1:
                        if status == "without helmet":
                            consecutive_no_helmet[s1_track_id] = (
                                consecutive_no_helmet.get(s1_track_id, 0) + 1
                            )

                            # 達到截圖門檻且尚未截圖過
                            if (
                                consecutive_no_helmet[s1_track_id]
                                >= screenshot_threshold
                                and s1_track_id not in captured_tracks
                            ):
                                captured_tracks.add(s1_track_id)

                                frame_rgb_snapshot = cv2.cvtColor(
                                    frame, cv2.COLOR_BGR2RGB
                                )
                                crop_rider_rgb = cv2.cvtColor(
                                    crop_s1, cv2.COLOR_BGR2RGB
                                )

                                timestamp = f"{current_frame / fps:.2f}s"
                                screenshot_data = {
                                    "track_id": s1_track_id,
                                    "frame": current_frame,
                                    "time": timestamp,
                                    "rider_img": crop_rider_rgb,
                                    "full_img": frame_rgb_snapshot,
                                }
                                screenshots_list.append(screenshot_data)

                                # 即時推播截圖到右側欄位
                                with violations_container:
                                    st.warning(
                                        f"🚨 騎士 #{s1_track_id} 連續"
                                        f" {screenshot_threshold} 影格未戴安全帽"
                                        f" (時間: {timestamp})"
                                    )
                                    st.image(
                                        crop_rider_rgb,
                                        caption=(
                                            f"騎士 #{s1_track_id} 局部特寫"
                                        ),
                                        use_container_width=True,
                                    )
                        else:
                            # 只有在「尚未被鎖定」為未戴安全帽時，出現 helmet 才會將連續計數歸零
                            if s1_track_id not in track_locked:
                                consecutive_no_helmet[s1_track_id] = 0

            # 轉 RGB 並渲染即時影像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

            if total_frames > 0:
                progress_bar.progress(min(current_frame / total_frames, 1.0))

        cap.release()
        os.remove(tfile.name)
        st.success("🎉 影片分析結束！")

        # ------------------------------------------------------------
        # 結算顯示所有截圖清單
        # ------------------------------------------------------------
        if len(screenshots_list) > 0:
            st.markdown("---")
            st.subheader(
                f"📋 違規總結報告（共捕捉到 {len(screenshots_list)} 起違規）"
            )
            cols = st.columns(3)
            for idx, item in enumerate(screenshots_list):
                with cols[idx % 3]:
                    st.image(
                        item["rider_img"],
                        caption=(
                            f"騎士 #{item['track_id']} | 時間:"
                            f" {item['time']} (第 {item['frame']} 影格)"
                        ),
                        use_container_width=True,
                    )
        else:
            st.info("✅ 本次影片未偵測到符合門檻的未戴安全帽違規行為。")
