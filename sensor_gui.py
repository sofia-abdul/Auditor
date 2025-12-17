import serial
import time
import numpy as np
import board
import busio
import adafruit_mlx90640
import matplotlib.pyplot as plt
import cv2
from matplotlib.animation import FuncAnimation
from email.message import EmailMessage
import smtplib
import os
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import customtkinter
from PIL import Image
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging

# ------------------ CONFIG ------------------ #

EMAIL_SENDER = 'ayunvuai@gmail.com'
EMAIL_PASSWORD = 'fejw blvu wkau alti'
EMAIL_RECEIVER = 'williejoseph299@gmail.com'

# Thresholds
TEMP_THRESHOLD = 32.0
RADAR_THRESHOLD = 2
EMAIL_COOLDOWN = 300
UPSCALE_FACTOR = 10
OUTPUT_DIR = "sensor_images"
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global state
state = {
    'start_time': time.time(),
    'warmup_time': 5,
    'radar_locked': False,
    'radar_stable_count': 0,
    'radar_threshold': RADAR_THRESHOLD,
    'last_alert_time': 0,
    'email_cooldown': EMAIL_COOLDOWN,
    'radar_buffer': [],
    'monitoring': False,
    'alert_count': 0,
    'last_detection_time': "None",
    'detection_count': 0
}

# Global plot data
timestamps = []
presence_data = []
thermal_data = np.zeros((24, 32))

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Auditor Monitoring System")
        self.geometry("1200x600")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.ser = None
        self.mlx = None
        self.thermal_frame = np.zeros((24 * 32,))
        self.show_main_app()

    def show_main_app(self):
        # Load images
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images")
        self.alerts_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "alerts.png")), size=(40, 40))
        self.sensor_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "flames.png")), size=(40, 40))
        self.detection_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "detection.png")), size=(50, 50))
        self.settings_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "system.png")), size=(40, 40))
        self.detection_count = customtkinter.CTkImage(Image.open(os.path.join(image_path, "detection_count.png")), size=(50, 50))


        # Create dashboard frame (full window)
        self.create_dashboard_frame()

    def create_dashboard_frame(self):
        self.dashboard_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.dashboard_frame.grid(row=0, column=0, sticky="nsew")
        self.dashboard_frame.grid_rowconfigure(0, weight=0)
        self.dashboard_frame.grid_rowconfigure(2, weight=3)
        self.dashboard_frame.grid_columnconfigure(0, weight=1)

        # Title
        label = customtkinter.CTkLabel(
            self.dashboard_frame, text="Auditor Monitoring Dashboard",
            font=customtkinter.CTkFont("Times New Roman", size=20, weight="bold")
        )
        label.grid(row=0, column=0, padx=20, pady=(10, 10), sticky="w")

        # Stats frame
        self.stats_frame = customtkinter.CTkFrame(self.dashboard_frame, height=100, corner_radius=10, fg_color="transparent")
        self.stats_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.stats_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Stat boxes
        self.create_stat_box(self.stats_frame, "Last Detection Time", "None", 0, image=self.detection_image)
        self.create_stat_box(self.stats_frame, "Detection Count", "0", 1, image=self.detection_count)
        self.create_stat_box(self.stats_frame, "Alerts", "0", 2, image=self.alerts_image)
        self.create_stat_box(self.stats_frame, "System", "Idle", 3, image=self.settings_image)

        # Bottom frame for plots
        bottom_frame = customtkinter.CTkFrame(self.dashboard_frame, corner_radius=10, fg_color="transparent")
        bottom_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        bottom_frame.grid_rowconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(0, weight=1)

        # Plot frame
        self.stats_chart_frame = customtkinter.CTkFrame(bottom_frame, corner_radius=10, fg_color="white")
        self.stats_chart_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.create_plots(self.stats_chart_frame)

        button_height = 35
        text_size = customtkinter.CTkFont("Times New Roman", size=15, weight='bold')

        # Control buttons
        self.controls_frame = customtkinter.CTkFrame(self.dashboard_frame, corner_radius=10, fg_color="transparent")
        self.controls_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.controls_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.start_button = customtkinter.CTkButton(
            self.controls_frame, text="Start Monitoring", command=self.start_monitoring, height=button_height, font=text_size
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.stop_button = customtkinter.CTkButton(
            self.controls_frame, text="Stop Monitoring", command=self.stop_monitoring, height=button_height, font=text_size
        )
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        self.email_button = customtkinter.CTkButton(
            self.controls_frame, text="Send Test Email", command=self.send_test_email, height=button_height, font=text_size
        )
        self.email_button.grid(row=0, column=2, padx=5, pady=5)

    def create_stat_box(self, parent, label, value, column, width=150, height=100, image=None):
        box = customtkinter.CTkFrame(parent, width=width, height=height, corner_radius=10, fg_color="white")
        box.grid(row=0, column=column, padx=5, pady=10, sticky="nsew")
        box.grid_columnconfigure(0, weight=1)
        box.grid_columnconfigure(1, weight=0)

        label_label = customtkinter.CTkLabel(
            box, text=label, font=customtkinter.CTkFont("Times New Roman", size=16, weight="bold")
        )
        label_label.grid(row=0, column=0, padx=10, pady=(8, 1), sticky="w")

        value_label = customtkinter.CTkLabel(
            box, text=value, font=customtkinter.CTkFont("Times New Roman", size=14)
        )
        value_label.grid(row=1, column=0, padx=10, pady=(0, 6), sticky="w")
        setattr(self, f"stat_{label.lower().replace(' ', '_')}", value_label)

        if image:
            icon = customtkinter.CTkLabel(box, text="", image=image)
            icon.grid(row=0, column=1, rowspan=2, padx=10, sticky="e")

        progress = customtkinter.CTkProgressBar(box)
        progress.grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 8), sticky="nsew")

    def create_plots(self, parent):
        #self.fig = plt.figure(figsize=(18, 5), gridspec_kw={'width_ratios': [1.2, 0.8, 1.5]})
        self.fig = plt.figure(figsize=(18, 5))
        # 3D Thermal plot
        self.ax3d = self.fig.add_subplot(131, projection='3d')
        x, y = np.meshgrid(np.arange(32), np.arange(24))
        z = np.zeros((24, 32))
        self.surf = self.ax3d.plot_surface(x, y, z, cmap='inferno', vmin=0, vmax=60, edgecolor='none')
        self.ax3d.set_xlabel('X (pixels)')
        self.ax3d.set_ylabel('Y (pixels)')
        self.ax3d.set_zlabel('Temperature (C)')
        self.ax3d.set_title("3D Thermal Image")
        #self.ax3d.set_frame_on(False)
        #cbar = self.fig.colorbar(self.surf, ax=self.ax3d)
        #cbar.set_label('Temperature [�C]', fontsize=12)

        # Radar plot
        self.radar_ax = self.fig.add_subplot(132)
        self.radar_line, = self.radar_ax.plot([], [], drawstyle='steps-post', linewidth=2, color='b')
        self.radar_ax.set_ylim(-0.2, 1.2)
        self.radar_ax.set_yticks([0, 1])
        self.radar_ax.set_yticklabels(["No Presence", "Presence"])
        self.radar_ax.set_xlabel("Time (s)")
        #self.radar_ax.set_ylabel("Presence")
        self.radar_ax.set_title("Radar Presence Detection")
        self.radar_ax.grid(True)

        # 2D Thermal plot
        self.thermal_ax = self.fig.add_subplot(133)
        self.thermal_img = self.thermal_ax.imshow(np.zeros((24, 32)), vmin=0, vmax=60, cmap='inferno', interpolation='bilinear')
        cbar = self.fig.colorbar(self.thermal_img, ax=self.thermal_ax)
        cbar.set_label('Temperature [C]', fontsize=12)
        self.thermal_ax.set_title("Thermal Image")
        #self.thermal_ax.set_xlabel('X (pixels)')
        #self.thermal_ax.set_ylabel('Y (pixels)')

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.fig.tight_layout()

    def setup_sensors(self):
        self.ser = setup_serial()
        if self.ser is None:
            logger.error("Failed to initialize radar sensor")
            return False
        self.mlx = setup_thermal()
        if self.mlx is None:
            logger.error("Failed to initialize thermal sensor")
            return False
        return True

    def update_plot(self, frame):
        global timestamps, presence_data, thermal_data
        now = time.time()
        radar_presence = read_presence_value(self.ser)

        if radar_presence is None:
            #logger.warning("Invalid radar data")
            return [self.radar_line, self.thermal_img, self.surf]

        # Warmup period
        if now - state['start_time'] < state['warmup_time']:
            state['radar_buffer'].append(radar_presence)
            if radar_presence == 1:
            
                state['last_detection_time'] = datetime.now().strftime("%H:%M:%S")
                state['detection_count'] += 1
                self.update_stat("last_detection_time", state['last_detection_time'])
                self.update_stat("detection_count", str(state['detection_count']))
                logger.info("Early radar detection during warmup")
            self.update_stat("system", "Warming up")
            return [self.radar_line, self.thermal_img, self.surf]

        # Radar detection

# Simplified: Update stats on every radar_presence == 1
        if radar_presence == 1:
            state['last_detection_time'] = datetime.now().strftime("%H:%M:%S")
            state['detection_count'] += 1
            self.update_stat("last_detection_time", state['last_detection_time'])
            self.update_stat("detection_count", str(state['detection_count']))
            logger.info("Human detected")
        else:
            state['no_presence_count'] = state.get('no_presence_count', 0) + 1
            logger.info("No human detected")

        # Keep radar_locked for email alerts to avoid spamming
        if radar_presence == 1:
            state['radar_stable_count'] += 1
            state['no_presence_count'] = 0
            if not state['radar_locked'] and state['radar_stable_count'] >= state['radar_threshold']:
                state['radar_locked'] = True
        else:
            if state['radar_locked'] and state['no_presence_count'] >= 4:
                state['radar_locked'] = False
                state['radar_stable_count'] = 0
                state['no_presence_count'] = 0

        # Update plots only if radar detects presence
        if state['radar_locked'] or radar_presence == 1:
            # Radar plot
            current_time = now - state['start_time']
            timestamps.append(current_time)
            presence_data.append(radar_presence)
            max_points = 100
            timestamps_trimmed = timestamps[-max_points:]
            presence_trimmed = presence_data[-max_points:]
            self.radar_line.set_data(timestamps_trimmed, presence_trimmed)
            self.radar_ax.relim()
            self.radar_ax.autoscale_view()

            # Thermal data
            thermal_data = read_thermal_value(self.mlx, self.thermal_frame)
            if thermal_data is not None:
                self.thermal_img.set_data(np.fliplr(thermal_data))
                self.thermal_img.set_clim(vmin=np.min(thermal_data), vmax=np.max(thermal_data))
                logger.info(f"Radar={radar_presence}, Max Temp={np.max(thermal_data):.1f}°C")
            else:
                thermal_data = np.zeros((24, 32))
                logger.warning("Thermal data unavailable")

            # 3D thermal plot
            self.ax3d.clear()
            x, y = np.meshgrid(np.arange(32), np.arange(24))
            z = thermal_data
            self.surf = self.ax3d.plot_surface(x, y, z, cmap='inferno', vmin=0, vmax=60, edgecolor='none')
            self.ax3d.set_xlabel('X (pixels)')
            self.ax3d.set_ylabel('Y (pixels)')
            self.ax3d.set_zlabel('Temperature (°C)')
            self.ax3d.set_title("3D Thermal Image")
            if thermal_data is not None:
                self.ax3d.set_zlim(np.min(thermal_data), np.max(thermal_data))

            # Save image and send email on new detection
            if state['radar_locked'] and now - state['last_alert_time'] > state['email_cooldown']:
                timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                plot_filename = os.path.join(OUTPUT_DIR, f"sensor_plot_{timestamp}.png")
                self.fig.savefig(plot_filename, bbox_inches='tight')
                logger.info(f"Saved plot: {plot_filename}")
                send_email_alert(plot_filename)
                state['last_alert_time'] = now
                state['alert_count'] += 1
                self.update_stat("alerts", str(state['alert_count']))

            self.canvas.draw()

        self.update_stat("system", "Monitoring" if state['monitoring'] else "Idle")
        return [self.radar_line, self.thermal_img, self.surf]

    def update_stat(self, key, value):
        label = getattr(self, f"stat_{key}", None)
        if label:
            label.configure(text=value)

    def sensor_thread(self):
        if not self.setup_sensors():
            self.update_stat("system", "Sensor Error")
            return
        state['monitoring'] = True
        self.animation = FuncAnimation(
            self.fig, self.update_plot, interval=100, blit=True, cache_frame_data=False
        )
        self.canvas.draw()
        logger.info("Sensor monitoring started")

    def start_monitoring(self):
        if not state['monitoring']:
            state['start_time'] = time.time()
            state['radar_buffer'].clear()
            state['alert_count'] = 0
            state['last_detection_time'] = "None"
            state['detection_count'] = 0
            state['radar_locked'] = False
            state['radar_stable_count'] = 0
            timestamps.clear()
            presence_data.clear()
            threading.Thread(target=self.sensor_thread, daemon=True).start()

    def stop_monitoring(self):
        state['monitoring'] = False
        if hasattr(self, 'animation'):
            self.animation.event_source.stop()
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
        self.update_stat("system", "Idle")
        logger.info("Monitoring stopped")

    def send_test_email(self):
        try:
            timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            plot_filename = os.path.join(OUTPUT_DIR, f"sensor_plot_{timestamp}.png")
            self.fig.savefig(plot_filename, bbox_inches='tight')
            send_email_alert(plot_filename)
            state['alert_count'] += 1
            self.update_stat("alerts", str(state['alert_count']))
            logger.info("Test email sent")
        except Exception as e:
            logger.error(f"Failed to send test email: {e}")

    def destroy(self):
        self.stop_monitoring()
        plt.close('all')
        super().destroy()

# Sensor functions
def setup_serial(port=SERIAL_PORT, baudrate=BAUD_RATE):
    try:
        ser = serial.Serial(port, baudrate, timeout=0.1)
        time.sleep(10)  # 10s stabilization as in non-GUI code
        while ser.in_waiting:
            ser.readline()
        logger.info(f"Connected to {port}")
        return ser
    except serial.SerialException as e:
        logger.error(f"Serial connection error: {e}")
        return None

def read_presence_value(ser):
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line in ["0", "1"]:
            return int(line)
        #logger.warning(f"Ignored invalid data: '{line}'")
        return None                                                                                                                                                    
    except Exception as e:
        logger.error(f"Serial read error: {e}")
        return None

def setup_thermal():
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        mlx = adafruit_mlx90640.MLX90640(i2c)
        mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
        logger.info("Thermal sensor initialized")
        return mlx
    except Exception as e:
        logger.error(f"Failed to initialize MLX90640: {e}")
        return None

def read_thermal_value(mlx, frame, max_retries=5):
    retry_count = 0
    while retry_count < max_retries:
        try:
            mlx.getFrame(frame)
            return np.reshape(frame, (24, 32))
        except (ValueError, RuntimeError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Thermal read failed after {max_retries} retries: {e}")
                return None
            time.sleep(0.1)
    return None

def send_email_alert(plot_path):
    try:
        msg = EmailMessage()
        msg['Subject'] = 'Human Detected - Sensor Alert'
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg.set_content(f'Human detected.\nSensor plot: {plot_path}')

        with open(plot_path, 'rb') as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='png', filename=os.path.basename(plot_path))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        logger.info(f"Email sent with {plot_path}")
    except Exception as e:
        logger.error(f"Email error: {e}")

if __name__ == "__main__":
    customtkinter.set_appearance_mode("System")
    app = App()
    app.mainloop()

import serial
import time
import numpy as np
import board
import busio
import adafruit_mlx90640
import matplotlib.pyplot as plt
import cv2
from matplotlib.animation import FuncAnimation
from email.message import EmailMessage
import smtplib
import os
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import customtkinter
from PIL import Image
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging

# ------------------ CONFIG ------------------ #

EMAIL_SENDER = 'ayunvuai@gmail.com'
EMAIL_PASSWORD = 'fejw blvu wkau alti'
EMAIL_RECEIVER = 'williejoseph299@gmail.com'

# Thresholds
TEMP_THRESHOLD = 32.0
RADAR_THRESHOLD = 2
EMAIL_COOLDOWN = 300
UPSCALE_FACTOR = 10
OUTPUT_DIR = "sensor_images"
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global state
state = {
    'start_time': time.time(),
    'warmup_time': 5,
    'radar_locked': False,
    'radar_stable_count': 0,
    'radar_threshold': RADAR_THRESHOLD,
    'last_alert_time': 0,
    'email_cooldown': EMAIL_COOLDOWN,
    'radar_buffer': [],
    'monitoring': False,
    'alert_count': 0,
    'last_detection_time': "None",
    'detection_count': 0
}

# Global plot data
timestamps = []
presence_data = []
thermal_data = np.zeros((24, 32))

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Auditor Monitoring System")
        self.geometry("1200x600")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.ser = None
        self.mlx = None
        self.thermal_frame = np.zeros((24 * 32,))
        self.show_main_app()

    def show_main_app(self):
        # Load images
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images")
        self.alerts_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "alerts.png")), size=(40, 40))
        self.sensor_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "flames.png")), size=(40, 40))
        self.detection_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "detection.png")), size=(50, 50))
        self.settings_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "system.png")), size=(40, 40))
        self.detection_count = customtkinter.CTkImage(Image.open(os.path.join(image_path, "detection_count.png")), size=(50, 50))


        # Create dashboard frame (full window)
        self.create_dashboard_frame()

    def create_dashboard_frame(self):
        self.dashboard_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.dashboard_frame.grid(row=0, column=0, sticky="nsew")
        self.dashboard_frame.grid_rowconfigure(0, weight=0)
        self.dashboard_frame.grid_rowconfigure(2, weight=3)
        self.dashboard_frame.grid_columnconfigure(0, weight=1)

        # Title
        label = customtkinter.CTkLabel(
            self.dashboard_frame, text="Auditor Monitoring Dashboard",
            font=customtkinter.CTkFont("Times New Roman", size=20, weight="bold")
        )
        label.grid(row=0, column=0, padx=20, pady=(10, 10), sticky="w")

        # Stats frame
        self.stats_frame = customtkinter.CTkFrame(self.dashboard_frame, height=100, corner_radius=10, fg_color="transparent")
        self.stats_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.stats_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Stat boxes
        self.create_stat_box(self.stats_frame, "Last Detection Time", "None", 0, image=self.detection_image)
        self.create_stat_box(self.stats_frame, "Detection Count", "0", 1, image=self.detection_count)
        self.create_stat_box(self.stats_frame, "Alerts", "0", 2, image=self.alerts_image)
        self.create_stat_box(self.stats_frame, "System", "Idle", 3, image=self.settings_image)

        # Bottom frame for plots
        bottom_frame = customtkinter.CTkFrame(self.dashboard_frame, corner_radius=10, fg_color="transparent")
        bottom_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        bottom_frame.grid_rowconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(0, weight=1)

        # Plot frame
        self.stats_chart_frame = customtkinter.CTkFrame(bottom_frame, corner_radius=10, fg_color="white")
        self.stats_chart_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.create_plots(self.stats_chart_frame)

        button_height = 35
        text_size = customtkinter.CTkFont("Times New Roman", size=15, weight='bold')

        # Control buttons
        self.controls_frame = customtkinter.CTkFrame(self.dashboard_frame, corner_radius=10, fg_color="transparent")
        self.controls_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.controls_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.start_button = customtkinter.CTkButton(
            self.controls_frame, text="Start Monitoring", command=self.start_monitoring, height=button_height, font=text_size
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.stop_button = customtkinter.CTkButton(
            self.controls_frame, text="Stop Monitoring", command=self.stop_monitoring, height=button_height, font=text_size
        )
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        self.email_button = customtkinter.CTkButton(
            self.controls_frame, text="Send Test Email", command=self.send_test_email, height=button_height, font=text_size
        )
        self.email_button.grid(row=0, column=2, padx=5, pady=5)

    def create_stat_box(self, parent, label, value, column, width=150, height=100, image=None):
        box = customtkinter.CTkFrame(parent, width=width, height=height, corner_radius=10, fg_color="white")
        box.grid(row=0, column=column, padx=5, pady=10, sticky="nsew")
        box.grid_columnconfigure(0, weight=1)
        box.grid_columnconfigure(1, weight=0)

        label_label = customtkinter.CTkLabel(
            box, text=label, font=customtkinter.CTkFont("Times New Roman", size=16, weight="bold")
        )
        label_label.grid(row=0, column=0, padx=10, pady=(8, 1), sticky="w")

        value_label = customtkinter.CTkLabel(
            box, text=value, font=customtkinter.CTkFont("Times New Roman", size=14)
        )
        value_label.grid(row=1, column=0, padx=10, pady=(0, 6), sticky="w")
        setattr(self, f"stat_{label.lower().replace(' ', '_')}", value_label)

        if image:
            icon = customtkinter.CTkLabel(box, text="", image=image)
            icon.grid(row=0, column=1, rowspan=2, padx=10, sticky="e")

        progress = customtkinter.CTkProgressBar(box)
        progress.grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 8), sticky="nsew")

    def create_plots(self, parent):
        #self.fig = plt.figure(figsize=(18, 5), gridspec_kw={'width_ratios': [1.2, 0.8, 1.5]})
        self.fig = plt.figure(figsize=(18, 5))
        # 3D Thermal plot
        self.ax3d = self.fig.add_subplot(131, projection='3d')
        x, y = np.meshgrid(np.arange(32), np.arange(24))
        z = np.zeros((24, 32))
        self.surf = self.ax3d.plot_surface(x, y, z, cmap='inferno', vmin=0, vmax=60, edgecolor='none')
        self.ax3d.set_xlabel('X (pixels)')
        self.ax3d.set_ylabel('Y (pixels)')
        self.ax3d.set_zlabel('Temperature (C)')
        self.ax3d.set_title("3D Thermal Image")
        #self.ax3d.set_frame_on(False)
        #cbar = self.fig.colorbar(self.surf, ax=self.ax3d)
        #cbar.set_label('Temperature [�C]', fontsize=12)

        # Radar plot
        self.radar_ax = self.fig.add_subplot(132)
        self.radar_line, = self.radar_ax.plot([], [], drawstyle='steps-post', linewidth=2, color='b')
        self.radar_ax.set_ylim(-0.2, 1.2)
        self.radar_ax.set_yticks([0, 1])
        self.radar_ax.set_yticklabels(["No Presence", "Presence"])
        self.radar_ax.set_xlabel("Time (s)")
        #self.radar_ax.set_ylabel("Presence")
        self.radar_ax.set_title("Radar Presence Detection")
        self.radar_ax.grid(True)

        # 2D Thermal plot
        self.thermal_ax = self.fig.add_subplot(133)
        self.thermal_img = self.thermal_ax.imshow(np.zeros((24, 32)), vmin=0, vmax=60, cmap='inferno', interpolation='bilinear')
        cbar = self.fig.colorbar(self.thermal_img, ax=self.thermal_ax)
        cbar.set_label('Temperature [C]', fontsize=12)
        self.thermal_ax.set_title("Thermal Image")
        #self.thermal_ax.set_xlabel('X (pixels)')
        #self.thermal_ax.set_ylabel('Y (pixels)')

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.fig.tight_layout()

    def setup_sensors(self):
        self.ser = setup_serial()
        if self.ser is None:
            logger.error("Failed to initialize radar sensor")
            return False
        self.mlx = setup_thermal()
        if self.mlx is None:
            logger.error("Failed to initialize thermal sensor")
            return False
        return True

    def update_plot(self, frame):
        global timestamps, presence_data, thermal_data
        now = time.time()
        radar_presence = read_presence_value(self.ser)

        if radar_presence is None:
            #logger.warning("Invalid radar data")
            return [self.radar_line, self.thermal_img, self.surf]

        # Warmup period
        if now - state['start_time'] < state['warmup_time']:
            state['radar_buffer'].append(radar_presence)
            if radar_presence == 1:
            
                state['last_detection_time'] = datetime.now().strftime("%H:%M:%S")
                state['detection_count'] += 1
                self.update_stat("last_detection_time", state['last_detection_time'])
                self.update_stat("detection_count", str(state['detection_count']))
                logger.info("Early radar detection during warmup")
            self.update_stat("system", "Warming up")
            return [self.radar_line, self.thermal_img, self.surf]

        # Radar detection

# Simplified: Update stats on every radar_presence == 1
        if radar_presence == 1:
            state['last_detection_time'] = datetime.now().strftime("%H:%M:%S")
            state['detection_count'] += 1
            self.update_stat("last_detection_time", state['last_detection_time'])
            self.update_stat("detection_count", str(state['detection_count']))
            logger.info("Human detected")
        else:
            state['no_presence_count'] = state.get('no_presence_count', 0) + 1
            logger.info("No human detected")

        # Keep radar_locked for email alerts to avoid spamming
        if radar_presence == 1:
            state['radar_stable_count'] += 1
            state['no_presence_count'] = 0
            if not state['radar_locked'] and state['radar_stable_count'] >= state['radar_threshold']:
                state['radar_locked'] = True
        else:
            if state['radar_locked'] and state['no_presence_count'] >= 4:
                state['radar_locked'] = False
                state['radar_stable_count'] = 0
                state['no_presence_count'] = 0

        # Update plots only if radar detects presence
        if state['radar_locked'] or radar_presence == 1:
            # Radar plot
            current_time = now - state['start_time']
            timestamps.append(current_time)
            presence_data.append(radar_presence)
            max_points = 100
            timestamps_trimmed = timestamps[-max_points:]
            presence_trimmed = presence_data[-max_points:]
            self.radar_line.set_data(timestamps_trimmed, presence_trimmed)
            self.radar_ax.relim()
            self.radar_ax.autoscale_view()

            # Thermal data
            thermal_data = read_thermal_value(self.mlx, self.thermal_frame)
            if thermal_data is not None:
                self.thermal_img.set_data(np.fliplr(thermal_data))
                self.thermal_img.set_clim(vmin=np.min(thermal_data), vmax=np.max(thermal_data))
                logger.info(f"Radar={radar_presence}, Max Temp={np.max(thermal_data):.1f}°C")
            else:
                thermal_data = np.zeros((24, 32))
                logger.warning("Thermal data unavailable")

            # 3D thermal plot
            self.ax3d.clear()
            x, y = np.meshgrid(np.arange(32), np.arange(24))
            z = thermal_data
            self.surf = self.ax3d.plot_surface(x, y, z, cmap='inferno', vmin=0, vmax=60, edgecolor='none')
            self.ax3d.set_xlabel('X (pixels)')
            self.ax3d.set_ylabel('Y (pixels)')
            self.ax3d.set_zlabel('Temperature (°C)')
            self.ax3d.set_title("3D Thermal Image")
            if thermal_data is not None:
                self.ax3d.set_zlim(np.min(thermal_data), np.max(thermal_data))

            # Save image and send email on new detection
            if state['radar_locked'] and now - state['last_alert_time'] > state['email_cooldown']:
                timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                plot_filename = os.path.join(OUTPUT_DIR, f"sensor_plot_{timestamp}.png")
                self.fig.savefig(plot_filename, bbox_inches='tight')
                logger.info(f"Saved plot: {plot_filename}")
                send_email_alert(plot_filename)
                state['last_alert_time'] = now
                state['alert_count'] += 1
                self.update_stat("alerts", str(state['alert_count']))

            self.canvas.draw()

        self.update_stat("system", "Monitoring" if state['monitoring'] else "Idle")
        return [self.radar_line, self.thermal_img, self.surf]

    def update_stat(self, key, value):
        label = getattr(self, f"stat_{key}", None)
        if label:
            label.configure(text=value)

    def sensor_thread(self):
        if not self.setup_sensors():
            self.update_stat("system", "Sensor Error")
            return
        state['monitoring'] = True
        self.animation = FuncAnimation(
            self.fig, self.update_plot, interval=100, blit=True, cache_frame_data=False
        )
        self.canvas.draw()
        logger.info("Sensor monitoring started")

    def start_monitoring(self):
        if not state['monitoring']:
            state['start_time'] = time.time()
            state['radar_buffer'].clear()
            state['alert_count'] = 0
            state['last_detection_time'] = "None"
            state['detection_count'] = 0
            state['radar_locked'] = False
            state['radar_stable_count'] = 0
            timestamps.clear()
            presence_data.clear()
            threading.Thread(target=self.sensor_thread, daemon=True).start()

    def stop_monitoring(self):
        state['monitoring'] = False
        if hasattr(self, 'animation'):
            self.animation.event_source.stop()
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
        self.update_stat("system", "Idle")
        logger.info("Monitoring stopped")

    def send_test_email(self):
        try:
            timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            plot_filename = os.path.join(OUTPUT_DIR, f"sensor_plot_{timestamp}.png")
            self.fig.savefig(plot_filename, bbox_inches='tight')
            send_email_alert(plot_filename)
            state['alert_count'] += 1
            self.update_stat("alerts", str(state['alert_count']))
            logger.info("Test email sent")
        except Exception as e:
            logger.error(f"Failed to send test email: {e}")

    def destroy(self):
        self.stop_monitoring()
        plt.close('all')
        super().destroy()

# Sensor functions
def setup_serial(port=SERIAL_PORT, baudrate=BAUD_RATE):
    try:
        ser = serial.Serial(port, baudrate, timeout=0.1)
        time.sleep(10)  # 10s stabilization as in non-GUI code
        while ser.in_waiting:
            ser.readline()
        logger.info(f"Connected to {port}")
        return ser
    except serial.SerialException as e:
        logger.error(f"Serial connection error: {e}")
        return None

def read_presence_value(ser):
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line in ["0", "1"]:
            return int(line)
        #logger.warning(f"Ignored invalid data: '{line}'")
        return None                                                                                                                                                    
    except Exception as e:
        logger.error(f"Serial read error: {e}")
        return None

def setup_thermal():
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        mlx = adafruit_mlx90640.MLX90640(i2c)
        mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
        logger.info("Thermal sensor initialized")
        return mlx
    except Exception as e:
        logger.error(f"Failed to initialize MLX90640: {e}")
        return None

def read_thermal_value(mlx, frame, max_retries=5):
    retry_count = 0
    while retry_count < max_retries:
        try:
            mlx.getFrame(frame)
            return np.reshape(frame, (24, 32))
        except (ValueError, RuntimeError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Thermal read failed after {max_retries} retries: {e}")
                return None
            time.sleep(0.1)
    return None

def send_email_alert(plot_path):
    try:
        msg = EmailMessage()
        msg['Subject'] = 'Human Detected - Sensor Alert'
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg.set_content(f'Human detected.\nSensor plot: {plot_path}')

        with open(plot_path, 'rb') as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='png', filename=os.path.basename(plot_path))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        logger.info(f"Email sent with {plot_path}")
    except Exception as e:
        logger.error(f"Email error: {e}")

if __name__ == "__main__":
    customtkinter.set_appearance_mode("System")
    app = App()
    app.mainloop()