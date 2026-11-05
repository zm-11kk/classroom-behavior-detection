# -*- coding: utf-8 -*-
"""
课堂学生行为检测系统 —— 增强版（8 分类）
基础功能不变：图片检测 / 视频检测 / 摄像头实时检测
优化点：
  1. 现代化 UI：深色渐变 + 玻璃拟态卡片 + 渐变标题
  2. 统计面板改为可视化进度条 + 概览卡片（人数 / 专注 / 互动 / 异常）
  3. 检测框中文标签（OpenCV 不支持中文，改用 PIL 绘制）
  4. 新增 IoU 阈值滑块
  5. 视频检测改为逐帧处理并输出带框视频（原 save=True 不会返回可播放文件）
  6. 摄像头增加实时 FPS 显示 + 累计统计简单去重
  7. 类别数自适应：模型类别数与列表不一致时自动回退，不会 IndexError
"""

import os
import time
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ==================== 配置区（只改这里） ====================
# 你的 8 个类别（显示顺序 = 模型输出 id 顺序 0~7）
CLASS_NAMES = [
    "低头写字",   # 0
    "低头看书",   # 1
    "抬头听课",   # 2
    "转头",       # 3
    "举手",       # 4
    "站立",       # 5
    "小组讨论",   # 6
    "教师指导",   # 7
]

# 分组概览卡片：按“模型原始英文类别名”归类（与中文显示名解耦，避免对不上导致 0%）
# train-16 模型的 12 类: Using_phone bend book bow_head hand-raising phone
#                        raise_head reading sleep turn_head upright writing
GROUPS = {
    "专注学习": ["bow_head", "reading", "book", "writing", "upright", "raise_head"],
    "课堂互动": ["hand-raising"],
    "异常行为": ["sleep", "Using_phone", "phone", "turn_head", "bend"],
}
# ==========================================================

# 直接加载模型（路径相对当前工作目录）
model = YOLO("runs/detect/train-16/weights/best.pt")

# 模型原始类别名（权威来源）
RAW_NAMES = [str(model.names[i]) for i in sorted(model.names.keys())]

# 英文类别名 -> 中文显示名（当前 12 类模型）
RAW_CN = {
    "Using_phone": "使用手机",
    "bend": "弯腰",
    "book": "看书",
    "bow_head": "低头",
    "hand-raising": "举手",
    "phone": "玩手机",
    "raise_head": "抬头",
    "reading": "读书",
    "sleep": "睡觉",
    "turn_head": "转头",
    "upright": "端坐",
    "writing": "写字",
}

# 显示名：类别数与 CLASS_NAMES 一致时用中文名列表，否则按英文映射转中文
if len(RAW_NAMES) == len(CLASS_NAMES):
    DISPLAY_NAMES = CLASS_NAMES
else:
    print(f"[提示] 模型类别数({len(RAW_NAMES)})与列表({len(CLASS_NAMES)})不一致，"
          f"已按模型类别自动显示。")
    DISPLAY_NAMES = [RAW_CN.get(n, n) for n in RAW_NAMES]
NUM_CLASSES = len(DISPLAY_NAMES)

# 每个类别一个醒目颜色（RGB）
PALETTE = [
    (56, 189, 248), (251, 146, 60), (52, 211, 153), (244, 114, 182),
    (167, 139, 250), (250, 204, 21), (248, 113, 113), (45, 212, 191),
    (96, 165, 250), (192, 132, 252), (74, 222, 128), (253, 186, 116),
]
def color_for(i):
    return PALETTE[i % len(PALETTE)]


# ==================== 全局状态 ====================
cap = None
conf_global = 0.30
iou_global = 0.45
total_count = [0] * NUM_CLASSES
last_seen = {}
fps_hist = []

def set_conf(v):
    global conf_global
    conf_global = float(v)

def set_iou(v):
    global iou_global
    iou_global = float(v)


# ==================== 绘制（含中文标签） ====================
_FONT = None
def get_font(size=20):
    global _FONT
    if _FONT is None:
        for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                   r"C:\Windows\Fonts\simsun.ttc"]:
            if os.path.exists(fp):
                try:
                    _FONT = ImageFont.truetype(fp, size)
                    break
                except Exception:
                    pass
        if _FONT is None:
            _FONT = ImageFont.load_default()
    return _FONT

def draw_boxes(frame, result):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = get_font()
    for box in result.boxes:
        cid = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        col = color_for(cid)
        draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
        label = f"{DISPLAY_NAMES[cid]} {conf:.2f}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y1 - th - 10 if y1 - th - 10 > 0 else y1 + 2
        draw.rectangle([x1, ty, x1 + tw + 12, ty + th + 10], fill=col)
        draw.text((x1 + 6, ty + 5), label, fill=(15, 23, 42), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def count_classes(result):
    cur = [0] * NUM_CLASSES
    for box in result.boxes:
        c = int(box.cls)
        if 0 <= c < NUM_CLASSES:
            cur[c] += 1
    return cur


# ==================== 统计面板 ====================
def build_stats_html(cur, fps=None):
    cur_total = sum(cur)

    # 分组概览卡片（按模型原始英文类别名匹配，统计才不会全是 0%）
    cards = ""
    if GROUPS:
        for gname, members in GROUPS.items():
            cnt = sum(cur[i] for i, n in enumerate(RAW_NAMES) if n in members)
            pct = (cnt / cur_total * 100) if cur_total else 0
            grad = {"专注学习": "linear-gradient(135deg,#10b981,#059669)",
                    "课堂互动": "linear-gradient(135deg,#0ea5e9,#2563eb)",
                    "异常行为": "linear-gradient(135deg,#f43f5e,#e11d48)"}.get(
                        gname, "linear-gradient(135deg,#6366f1,#8b5cf6)")
            cards += f"""
            <div style="background:{grad};border-radius:14px;padding:12px;text-align:center;color:#fff;
                        box-shadow:0 6px 20px rgba(0,0,0,.35);">
              <div style="font-size:26px;font-weight:800;line-height:1.1;">{cnt}</div>
              <div style="font-size:12px;opacity:.92;margin-top:2px;">{gname} {pct:.0f}%</div>
            </div>"""

    rows = ""
    maxc = max(cur) if cur_total else 1
    for i, name in enumerate(DISPLAY_NAMES):
        c = cur[i]
        w = (c / maxc * 100) if maxc else 0
        col = color_for(i)
        col = f"rgb({col[0]},{col[1]},{col[2]})"
        rows += f"""
        <div style="display:flex;align-items:center;gap:10px;margin:7px 0;font-size:13px;">
          <span style="width:78px;color:#cbd5e1;">{name}</span>
          <div style="flex:1;background:rgba(255,255,255,.08);border-radius:8px;height:16px;overflow:hidden;">
            <div style="width:{w:.1f}%;height:100%;background:{col};border-radius:8px;
                        box-shadow:0 0 10px {col};transition:width .3s ease;"></div>
          </div>
          <span style="width:30px;text-align:right;color:#f1f5f9;font-weight:700;">{c}</span>
        </div>"""

    fps_txt = f'<span style="color:#38bdf8;font-weight:600;">{fps:.1f} FPS</span>' if fps else ""
    ncol = max(len(GROUPS), 1) + 1
    return f"""
    <div style="font-family:'Segoe UI',system-ui,sans-serif;">
      <div style="display:grid;grid-template-columns:repeat({ncol},1fr);gap:10px;margin-bottom:6px;">
        <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);border-radius:14px;padding:12px;
                    text-align:center;color:#fff;box-shadow:0 6px 20px rgba(0,0,0,.35);">
          <div style="font-size:26px;font-weight:800;line-height:1.1;">{cur_total}</div>
          <div style="font-size:12px;opacity:.92;margin-top:2px;">画面人数</div>
        </div>
        {cards}
      </div>
      <div style="color:#94a3b8;font-size:12px;margin:14px 0 6px;">
        实时行为分布 {fps_txt}
      </div>
      {rows}
    </div>
    """


# ==================== 图片检测 ====================
def detect_image(img):
    if img is None:
        return None, "<div style='color:#94a3b8'>请先上传图片</div>"
    res = model(img, conf=conf_global, iou=iou_global, verbose=False)[0]
    annotated = draw_boxes(res.orig_img, res)
    cur = count_classes(res)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), build_stats_html(cur)


# ==================== 视频检测（逐帧处理后输出可播放视频） ====================
def detect_video(video_path, progress=gr.Progress()):
    if not video_path:
        return None, "<div style='color:#94a3b8'>请先上传视频</div>"
    vc = cv2.VideoCapture(video_path)
    if not vc.isOpened():
        return None, "<div style='color:#f43f5e'>无法打开视频文件</div>"

    total = int(vc.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = vc.get(cv2.CAP_PROP_FPS) or 25
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs("outputs", exist_ok=True)
    # 用绝对路径，避免 Gradio 因工作目录不同而找不到输出文件
    out_path = os.path.abspath(os.path.join("outputs", f"detect_{int(time.time())}.mp4"))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    agg = [0] * NUM_CLASSES
    idx = 0
    while True:
        ret, frame = vc.read()
        if not ret:
            break
        idx += 1
        res = model(frame, conf=conf_global, iou=iou_global, verbose=False)[0]
        writer.write(draw_boxes(frame, res))
        for i, c in enumerate(count_classes(res)):
            agg[i] += c
        progress(idx / total, desc=f"处理中 {idx}/{total}")
    vc.release()
    writer.release()

    # 统计面板显示“平均每帧人数”（用帧累计值除以帧数），避免一直显示 0
    frames = max(idx, 1)
    avg = [round(a / frames) for a in agg]
    return out_path, build_stats_html(avg)


def clear_video_result():
    """视频被删除/替换时清空上一次的检测结果。"""
    return None, build_stats_html([0] * NUM_CLASSES)


# ==================== 摄像头检测 ====================
def get_camera_frame():
    global last_seen, fps_hist
    if cap is None or not cap.isOpened():
        return None, "<div style='color:#94a3b8'>摄像头未启动</div>"

    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        return None, "<div style='color:#f43f5e'>读取画面失败</div>"

    res = model(frame, conf=conf_global, iou=iou_global, verbose=False)[0]
    annotated = draw_boxes(frame, res)
    cur = count_classes(res)

    now = time.time()
    for box in res.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        key = (int((x1 + x2) / 2) // 20, int((y1 + y2) / 2) // 20, int(box.cls))
        if key not in last_seen or now - last_seen[key] > 2.0:
            total_count[int(box.cls)] += 1
        last_seen[key] = now
    last_seen = {k: v for k, v in last_seen.items() if now - v < 3.0}

    dt = time.time() - t0
    fps_hist.append(1 / dt if dt > 0 else 0)
    if len(fps_hist) > 20:
        fps_hist.pop(0)
    fps = sum(fps_hist) / len(fps_hist)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), build_stats_html(cur, fps)

def start_cam():
    global cap, total_count, last_seen, fps_hist
    total_count = [0] * NUM_CLASSES
    last_seen = {}
    fps_hist = []
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "<div style='color:#f43f5e'>打开摄像头失败，请检查权限或设备占用</div>"
    return get_camera_frame()

def stop_cam():
    global cap
    if cap:
        cap.release()
        cap = None
    return None, "<div style='color:#94a3b8'>摄像头已停止</div>"


# ==================== 界面 ====================
CSS = """
body, .gradio-container {
    background: radial-gradient(circle at 20% 0%, #1e293b 0%, #0f172a 45%, #020617 100%) !important;
    font-family:'Segoe UI',system-ui,-apple-system,sans-serif !important;
}
.gradio-container { max-width:1440px !important; margin:0 auto !important; }
#title-bar {
    background: linear-gradient(90deg,#38bdf8,#818cf8,#c084fc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    font-size:34px; font-weight:800; text-align:center;
    padding:22px 0 4px 0; letter-spacing:1px;
}
#subtitle { text-align:center; color:#94a3b8; font-size:14px; margin-bottom:18px; }
.glass {
    background: rgba(30,41,59,.66) !important;
    border:1px solid rgba(148,163,184,.18) !important;
    border-radius:18px !important;
    backdrop-filter: blur(14px) !important;
    box-shadow:0 10px 40px rgba(0,0,0,.4) !important;
    padding:16px !important;
}
.tab-nav button { color:#cbd5e1 !important; font-weight:600 !important; }
.tab-nav button.selected { color:#38bdf8 !important; border-bottom:2px solid #38bdf8 !important; }
footer { display:none !important; }
"""

theme = gr.themes.Soft(primary_hue="sky", secondary_hue="indigo", neutral_hue="slate").set(
    body_background_fill="transparent",
    block_background_fill="rgba(30,41,59,.65)",
    block_border_color="rgba(148,163,184,.2)",
    input_background_fill="rgba(15,23,42,.6)",
    body_text_color="#e2e8f0",
)

with gr.Blocks(title="课堂学生行为检测系统", theme=theme, css=CSS) as demo:
    gr.HTML('<div id="title-bar">课堂学生行为智能检测系统</div>'
            '<div id="subtitle">YOLOv8 · 实时课堂行为识别与统计</div>')

    with gr.Row():
        conf_slider = gr.Slider(0.10, 0.90, value=0.30, step=0.05,
                                label="置信度阈值 (conf)", info="越高越严格，误检更少")
        iou_slider = gr.Slider(0.10, 0.90, value=0.45, step=0.05,
                               label="IoU 阈值", info="控制重叠框合并程度")
    conf_slider.change(set_conf, inputs=conf_slider)
    iou_slider.change(set_iou, inputs=iou_slider)

    empty = build_stats_html([0] * NUM_CLASSES)

    with gr.Tab("图片检测"):
        with gr.Row():
            with gr.Column(scale=1, elem_classes="glass"):
                img_in = gr.Image(label="上传图片", type="numpy", height=400)
                btn_img = gr.Button("开始检测", variant="primary", size="lg")
            with gr.Column(scale=1, elem_classes="glass"):
                img_out = gr.Image(label="检测结果", height=400)
                img_stats = gr.HTML(empty)
        btn_img.click(detect_image, inputs=img_in, outputs=[img_out, img_stats])
        # 图片一有变化（上传新图 / 删除）就自动更新检测结果
        img_in.change(detect_image, inputs=img_in, outputs=[img_out, img_stats])

    with gr.Tab("视频检测"):
        with gr.Row():
            with gr.Column(scale=1, elem_classes="glass"):
                vid_in = gr.Video(label="上传视频")
                btn_vid = gr.Button("开始检测", variant="primary", size="lg")
            with gr.Column(scale=1, elem_classes="glass"):
                vid_out = gr.Video(label="输出视频")
                vid_stats = gr.HTML(empty)
        btn_vid.click(detect_video, inputs=vid_in, outputs=[vid_out, vid_stats])
        # 视频被删除/替换时清空上一次结果，防止旧结果残留
        vid_in.change(clear_video_result, inputs=None, outputs=[vid_out, vid_stats])

    with gr.Tab("摄像头实时检测"):
        with gr.Row():
            with gr.Column(scale=2, elem_classes="glass"):
                cam_out = gr.Image(label="摄像头画面", height=440)
                with gr.Row():
                    cam_btn = gr.Button("启动摄像头", variant="primary")
                    stop_btn = gr.Button("停止", variant="stop")
            with gr.Column(scale=1, elem_classes="glass"):
                info_html = gr.HTML(empty)
        timer = gr.Timer(value=0.1, active=False)
        # concurrency_limit=1：避免上一帧还没推理完就排队下一帧，导致画面越来越卡
        timer.tick(get_camera_frame, outputs=[cam_out, info_html], concurrency_limit=1)
        cam_btn.click(start_cam, outputs=[cam_out, info_html]).then(
            lambda: gr.Timer(active=True), outputs=timer)
        stop_btn.click(stop_cam, outputs=[cam_out, info_html]).then(
            lambda: gr.Timer(active=False), outputs=timer)
        # 关闭页面时自动释放摄像头，避免下次启动提示“设备被占用”
        demo.unload(stop_cam)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
