# -*- coding: utf-8 -*-
"""
ESP32-CAM → Python 스쿼트 카운터 → Unity UDP 텔레메트리
최종 최적화 버전 (캘리브레이션 + 안정화)
"""

import cv2, time, numpy as np, argparse, threading, socket, json

# ==================== UDP 텔레메트리 ====================
class UDPTelemetry:
    """Unity로 텔레메트리 전송"""
    def __init__(self, host='127.0.0.1', port=9999):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (host, port)
        print(f"[UDP] 송신 준비 → {host}:{port}")
    
    def send(self, data_dict):
        try:
            message = json.dumps(data_dict, ensure_ascii=False).encode('utf-8')
            self.sock.sendto(message, self.addr)
        except Exception as e:
            print(f"[UDP] 전송 오류: {e}")
    
    def close(self):
        self.sock.close()

# ==================== 각도 및 깊이 계산 ====================
def estimate_knee_angle(rel_height_change):
    """높이 변화율 → 무릎 각도 추정"""
    angle = 180 - (rel_height_change * 450)
    return max(60, min(180, angle))

def calculate_depth_percent(angle):
    """각도 → 깊이 퍼센트"""
    if angle >= 170:
        return 0
    depth = ((180 - angle) / 90) * 100
    return min(100, max(0, depth))

def check_form_quality(depth_percent):
    """폼 평가"""
    if depth_percent < 50:
        return "TOO_SHALLOW", 50 - depth_percent
    elif depth_percent >= 85:
        return "GOOD", 0
    else:
        return "MODERATE", 85 - depth_percent

# ==================== 인수 파싱 ====================
parser = argparse.ArgumentParser()
parser.add_argument('--source', default='0')
parser.add_argument('--tracker', choices=['CSRT','KCF'], default='KCF')
parser.add_argument('--ema', type=float, default=0.30)  # 0.25 → 0.30 (평활화 강화)
parser.add_argument('--drop', type=float, default=0.20)  # 0.15 → 0.20 (더 깊이 앉아야 DOWN)
parser.add_argument('--rise', type=float, default=0.15)  # 0.10 → 0.15 (UP 판정 완화)
parser.add_argument('--detect-every', type=int, default=15)  # 10 → 15 (재검출 간격 증가)
parser.add_argument('--roi', default='0.2,0.1,0.8,0.9')
parser.add_argument('--hold', type=int, default=7)  # 3 → 7 (상태 안정화)
parser.add_argument('--calibration', type=int, default=10)  # 캘리브레이션 시간 (초)
parser.add_argument('--fallback-mjpeg', action='store_true')
parser.add_argument('--udp-host', default='127.0.0.1')
parser.add_argument('--udp-port', type=int, default=9999)
args = parser.parse_args()

x0, y0, x1, y1 = map(float, args.roi.split(','))
ROI = (x0, y0, x1, y1)

# ==================== 스레드 카메라 ====================
class ThreadedCamera:
    def __init__(self, source, fallback_mjpeg=False):
        self.source = source
        self.fallback = fallback_mjpeg
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        
        if isinstance(source, str) and source.startswith('http') and fallback_mjpeg:
            self._init_mjpeg()
        else:
            self._init_opencv()
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
    
    def _init_opencv(self):
        src = 0 if self.source == '0' else self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"소스를 열 수 없음: {self.source}")
        self.mode = 'opencv'
        print(f"[Camera] OpenCV 모드: {self.source}")
    
    def _init_mjpeg(self):
        import requests
        self.stream = requests.get(self.source, stream=True, timeout=10)
        self.buffer = b''
        self.mode = 'mjpeg'
        print(f"[Camera] MJPEG 모드: {self.source}")
    
    def _update(self):
        while not self.stopped:
            if self.mode == 'opencv':
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
            else:
                self._read_mjpeg()
    
    def _read_mjpeg(self):
        try:
            for chunk in self.stream.iter_content(chunk_size=4096):
                if self.stopped:
                    break
                self.buffer += chunk
                a = self.buffer.find(b'\xff\xd8')
                b = self.buffer.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = self.buffer[a:b+2]
                    self.buffer = self.buffer[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.lock:
                            self.frame = frame
        except Exception as e:
            print(f"[Camera] MJPEG 오류: {e}")
    
    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.stopped = True
        self.thread.join()
        if self.mode == 'opencv':
            self.cap.release()

# ==================== HOG 검출 ====================
def detect_person(frame, roi_box):
    x1, y1, x2, y2 = roi_box
    roi = frame[y1:y2, x1:x2]
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(roi, winStride=(8,8), padding=(4,4), scale=1.05)
    
    if len(boxes) == 0:
        return None
    
    largest = max(boxes, key=lambda b: b[2]*b[3])
    x, y, w, h = largest
    return (x1+x, y1+y, w, h)

def create_tracker(name):
    """OpenCV 버전 호환 Tracker 생성"""
    try:
        # OpenCV 4.5.1+ (legacy 모듈)
        if name == 'CSRT':
            return cv2.legacy.TrackerCSRT_create()
        elif name == 'KCF':
            return cv2.legacy.TrackerKCF_create()
        else:
            raise ValueError(f"알 수 없는 트래커: {name}")
    except AttributeError:
        # OpenCV 4.5.0 이하 (구버전 API)
        if name == 'CSRT':
            return cv2.TrackerCSRT_create()
        elif name == 'KCF':
            return cv2.TrackerKCF_create()
        else:
            raise ValueError(f"알 수 없는 트래커: {name}")

# ==================== 메인 ====================
def main():
    udp = UDPTelemetry(host=args.udp_host, port=args.udp_port)
    cam = ThreadedCamera(args.source, fallback_mjpeg=args.fallback_mjpeg)
    time.sleep(1)
    
    tracker = None
    bbox = None
    baseline_h = None
    smoothed_h = None
    count = 0
    state = "UP"
    hold_count = 0
    frame_idx = 0
    fps_list = []
    
    # 캘리브레이션 변수
    calibration_samples = []
    calibration_start = None
    calibration_complete = False
    
    print("\n[Start] 스쿼트 카운터 + UDP 텔레메트리")
    print(f"[설정] DROP={args.drop}, RISE={args.rise}, HOLD={args.hold}, EMA={args.ema}")
    print("- 'q': 종료, 'r': 리셋\n")
    
    try:
        while True:
            t_start = time.time()
            frame = cam.read()
            
            if frame is None:
                time.sleep(0.1)
                continue
            
            h, w = frame.shape[:2]
            roi_x1 = int(w * ROI[0])
            roi_y1 = int(h * ROI[1])
            roi_x2 = int(w * ROI[2])
            roi_y2 = int(h * ROI[3])
            roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)
            
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255,255,0), 2)
            
            if tracker is None or frame_idx % args.detect_every == 0:
                result = detect_person(frame, roi_box)
                if result is not None:
                    x, y, w_bb, h_bb = result
                    bbox = (x, y, w_bb, h_bb)
                    tracker = create_tracker(args.tracker)
                    tracker.init(frame, bbox)
                    
                    # 캘리브레이션 시작
                    if baseline_h is None and not calibration_complete:
                        if calibration_start is None:
                            calibration_start = time.time()
                            print(f"[캘리브레이션] 시작 - {args.calibration}초 동안 가만히 서 있으세요...")
            
            # 텔레메트리 기본값
            knee_angle = 180.0
            depth_percent = 0.0
            form_status = "IDLE"
            shortage = 0.0
            
            if tracker is not None:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w_bb, h_bb = [int(v) for v in bbox]
                    
                    # 캘리브레이션 진행
                    if baseline_h is None and calibration_start is not None:
                        elapsed = time.time() - calibration_start
                        calibration_samples.append(h_bb)
                        
                        # 진행 상황 표시
                        progress = min(int((elapsed / args.calibration) * 100), 100)
                        bar_width = int(w * 0.6)
                        bar_height = 30
                        bar_x = (w - bar_width) // 2
                        bar_y = h - 80
                        
                        # 진행 바 배경
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                                      (50, 50, 50), -1)
                        # 진행 바 채우기
                        fill_width = int(bar_width * progress / 100)
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                                      (0, 255, 0), -1)
                        # 테두리
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                                      (255, 255, 255), 2)
                        # 텍스트
                        cv2.putText(frame, f"CALIBRATING... {progress}%", 
                                    (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 255, 255), 2)
                        
                        remaining = max(0, args.calibration - int(elapsed))
                        cv2.putText(frame, f"{remaining}s", 
                                    (bar_x + bar_width + 10, bar_y + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # 캘리브레이션 완료
                        if elapsed >= args.calibration:
                            # 중간값 사용 (이상치 제거)
                            sorted_samples = sorted(calibration_samples)
                            mid_start = len(sorted_samples) // 4
                            mid_end = mid_start * 3
                            baseline_h = np.mean(sorted_samples[mid_start:mid_end])
                            smoothed_h = baseline_h
                            calibration_complete = True
                            
                            print(f"[캘리브레이션] 완료!")
                            print(f"[Baseline] 높이: {baseline_h:.1f} (샘플: {len(calibration_samples)}개)")
                            print(f"[시작] 스쿼트를 시작하세요!\n")
                            
                            # 완료 메시지 표시
                            cv2.putText(frame, "CALIBRATION COMPLETE!", 
                                        ((w - 500) // 2, h // 2), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                            cv2.imshow('Squat Counter', frame)
                            cv2.waitKey(2000)  # 2초 대기
                        
                        cv2.rectangle(frame, (x, y), (x+w_bb, y+h_bb), (255, 255, 0), 2)
                    
                    # 캘리브레이션 완료 후 정상 로직
                    elif baseline_h is not None:
                        smoothed_h = args.ema * h_bb + (1 - args.ema) * smoothed_h
                        rel = (baseline_h - smoothed_h) / baseline_h
                        
                        knee_angle = estimate_knee_angle(rel)
                        depth_percent = calculate_depth_percent(knee_angle)
                        form_status, shortage = check_form_quality(depth_percent)
                        
                        if state == "UP" and rel > args.drop:
                            hold_count += 1
                            if hold_count >= args.hold:
                                state = "DOWN"
                                hold_count = 0
                                print(f"[DOWN] 각도={knee_angle:.0f}° 깊이={depth_percent:.0f}%")
                        elif state == "DOWN" and rel < args.rise:
                            hold_count += 1
                            if hold_count >= args.hold:
                                state = "UP"
                                count += 1
                                hold_count = 0
                                print(f"[UP] 카운트: {count}")
                        else:
                            hold_count = 0
                        
                        color = (0,255,0) if state == "DOWN" else (0,255,255)
                        cv2.rectangle(frame, (x, y), (x+w_bb, y+h_bb), color, 2)
                        
                        # HUD
                        cv2.putText(frame, f"COUNT: {count}", (12, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
                        cv2.putText(frame, f"STATE: {state}", (12, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, f"ANGLE: {knee_angle:.0f}deg  DEPTH: {depth_percent:.0f}%", 
                                    (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                        form_color = (0,255,0) if form_status == "GOOD" else (0,165,255)
                        cv2.putText(frame, form_status, (12, 120), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, form_color, 2)
                else:
                    tracker = None
            
            # UDP 전송 (15Hz)
            if frame_idx % 2 == 0 and calibration_complete:
                fps_avg = np.mean(fps_list[-30:]) if fps_list else 0
                telemetry_data = {
                    "timestamp": time.time(),
                    "count": count,
                    "state": state,
                    "knee_angle": round(knee_angle, 1),
                    "depth_percent": round(depth_percent, 1),
                    "form_status": form_status,
                    "shortage_percent": round(shortage, 1),
                    "fps": round(fps_avg, 1)
                }
                udp.send(telemetry_data)
            
            # FPS
            elapsed = time.time() - t_start
            if elapsed > 0:
                fps_list.append(1.0 / elapsed)
            if len(fps_list) > 30:
                fps_list.pop(0)
            fps_avg = np.mean(fps_list) if fps_list else 0
            cv2.putText(frame, f"FPS: {fps_avg:.1f}", (w-150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            cv2.imshow('Squat Counter', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracker = None
                baseline_h = None
                calibration_start = None
                calibration_complete = False
                calibration_samples = []
                count = 0
                print("[Reset]")
            
            frame_idx += 1
    
    except KeyboardInterrupt:
        print("\n[Stop]")
    finally:
        udp.close()
        cam.stop()
        cv2.destroyAllWindows()
        print("[End]")

if __name__ == '__main__':
    main()
