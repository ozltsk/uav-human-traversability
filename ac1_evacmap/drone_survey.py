"""Модуль керування обльотом DJI Tello.

Дрон злітає біля входу, піднімається на задану висоту і проходить точки
зависання, задані оператором над ділянками чистої підлоги. У кожній точці
виконує чотири повороти на 90 градусів і в кожному напрямку записує кадр,
висоту за ToF-датчиком і кут рискання з IMU. Положення точки визначається
за виконаними командами переміщення (числення шляху), які Tello відпрацьовує
за оптичним потоком нижньої камери. Результат -- журнал обльоту FlightLog.
"""
import csv
import time

import numpy as np

from flight_log import FlightLog, FrameRecord


def load_waypoints(path):
    """CSV з колонками x, y (метри, відносно точки зльоту; x вперед, y вліво)."""
    pts = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pts.append((float(row["x"]), float(row["y"])))
    return pts


def grid_waypoints(x_range, y_range, spacing):
    """Регулярна сітка точок зависання ("змійкою")."""
    xs = np.arange(x_range[0], x_range[1] + 1e-6, spacing)
    ys = np.arange(y_range[0], y_range[1] + 1e-6, spacing)
    pts = []
    for i, y in enumerate(ys):
        row = [(float(x), float(y)) for x in xs]
        pts.extend(row if i % 2 == 0 else row[::-1])
    return pts


class TelloSurvey:
    def __init__(self, out_dir, height=1.3, speed_cm_s=30, settle_s=1.5, frame_size=(960, 720)):
        self.out_dir = out_dir
        self.height = height
        self.speed = speed_cm_s
        self.settle_s = settle_s
        self.frame_size = frame_size
        self.tello = None
        self.reader = None
        self.position = np.zeros(2)       # числення шляху, м
        self.log = FlightLog(root=out_dir)

    # ---------- зв'язок ----------
    def connect(self):
        from djitellopy import Tello
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        self.reader = self.tello.get_frame_read()
        time.sleep(2.0)
        print(f"Tello: батарея {self.tello.get_battery()} %")

    def disconnect(self):
        if self.tello is not None:
            try:
                self.tello.streamoff()
            finally:
                self.tello.end()

    # ---------- телеметрія ----------
    def read_tof_m(self):
        """Висота над підлогою за ToF-датчиком, м."""
        return self.tello.get_distance_tof() / 100.0

    def read_yaw(self):
        return float(self.tello.get_yaw())

    def grab_frame(self):
        frame = self.reader.frame                      # BGR від OpenCV
        return np.ascontiguousarray(frame[:, :, ::-1])  # -> RGB

    # ---------- переміщення ----------
    def move_to(self, x, y):
        """Переміщення до точки (x, y) відносно зльоту командами Tello (см)."""
        dx, dy = x - self.position[0], y - self.position[1]
        dx_cm, dy_cm = int(round(dx * 100)), int(round(dy * 100))
        if abs(dx_cm) >= 20 or abs(dy_cm) >= 20:
            self.tello.go_xyz_speed(max(-500, min(500, dx_cm)), max(-500, min(500, dy_cm)), 0, self.speed)
        self.position = np.array([x, y], dtype=np.float64)
        time.sleep(self.settle_s)

    def climb_to(self, height):
        current = self.read_tof_m()
        delta_cm = int(round((height - current) * 100))
        if delta_cm >= 20:
            self.tello.move_up(min(delta_cm, 500))
        elif delta_cm <= -20:
            self.tello.move_down(min(-delta_cm, 500))
        time.sleep(self.settle_s)

    # ---------- обліт ----------
    def survey(self, waypoints, turns=4):
        self.log.prepare()
        self.log.meta = {"source": "tello", "frame_size": list(self.frame_size),
                         "hfov_deg": 70.0, "tilt_deg": 10.0, "height": self.height,
                         "waypoints": [list(p) for p in waypoints]}
        index = 0
        try:
            self.tello.takeoff()
            time.sleep(self.settle_s)
            self.climb_to(self.height)
            yaw0 = self.read_yaw()
            for wi, (x, y) in enumerate(waypoints):
                self.move_to(x, y)
                for k in range(turns):
                    time.sleep(self.settle_s)
                    frame = self.grab_frame()
                    rec = FrameRecord(index=index, x=float(self.position[0]), y=float(self.position[1]),
                                      z_tof=self.read_tof_m(), yaw_deg=self.read_yaw() - yaw0, waypoint=wi)
                    self.log.add_frame(rec, image=frame)
                    print(f"точка {wi:2d}, кадр {index:4d}: висота {rec.z_tof:.2f} м, "
                          f"рискання {rec.yaw_deg:6.1f} град")
                    index += 1
                    if k < turns - 1:
                        self.tello.rotate_clockwise(90)
                self.tello.rotate_clockwise(90)        # повернення в початкову орієнтацію
        finally:
            try:
                self.tello.land()
            except Exception as exc:  # дрон уже на підлозі або зв'язок втрачено
                print("посадка:", exc)
            self.log.save()
        return self.log
