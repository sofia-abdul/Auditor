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

# ------------------ CONFIG ------------------ #

EMAIL_SENDER = 'ayunvuai@gmail.com'
EMAIL_PASSWORD = 'fejw blvu wkau alti'  # Use os.environ for production
EMAIL_RECEIVER = 'williejoseph299@gmail.com'

# Thresholds
TEMP_THRESHOLD = 32.0  # Celsius, for logging only
RADAR_THRESHOLD = 2    # Consecutive 1s for radar detection
EMAIL_COOLDOWN = 300   # 5 minutes between alerts
UPSCALE_FACTOR = 10    # Thermal image: 24x32 -> 240x320

# Output directory
OUTPUT_DIR = "sensor_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global lists for radar data
timestamps = []
presence_data = []

# ------------------ EMAIL FUNCTION ------------------ #

def send_email_alert(thermal_path, radar_path):
    """Send email with thermal and radar images."""
    try:
        msg = EmailMessage()
        msg['Subject'] = 'Human Detected - Radar & Thermal Alert'
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg.set_content(f'Human detected.\nThermal image: {thermal_path}\nRadar plot: {radar_path}')

        # Attach thermal image
        with open(thermal_path, 'rb') as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='png', filename=os.path.basename(thermal_path))

        # Attach radar plot
        with open(radar_path, 'rb') as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='png', filename=os.path.basename(radar_path))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"Email sent with {thermal_path} and {radar_path}")
    except Exception as e:
        print(f"[Email Error] {e}")

# ------------------ RADAR SENSOR SETUP ------------------ #

def setup_serial(port="/dev/ttyACM0", baudrate=115200):
    """Initialize and return the serial connection."""
    try:
        ser = serial.Serial(port, baudrate, timeout=0.1)
        time.sleep(10)  # 10s stabilization
        print(f"Connected to {port} at {baudrate} baud")
        while ser.in_waiting:
            ser.readline()
        return ser
    except serial.SerialException as e:
        print(f"Serial connection error: {e}")
        return None

def read_presence_value(ser):
    """Read and clean a line from serial, returning 0 or 1 if valid."""
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        #print(f"[Debug] Raw serial data: '{line}'")
        if line == "0" or line == "1":
            return int(line)
        else:
            #print(f"[Radar Warning] Ignored invalid data: '{line}'")
            return None
    except Exception as e:
        print(f"[Radar Error] Serial read error: {e}")
        return None
    

# ------------------ THERMAL SENSOR SETUP ------------------ #

def setup_thermal():
    """Initialize and return the MLX90640 thermal sensor."""
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        mlx = adafruit_mlx90640.MLX90640(i2c)
        mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
        print(f"Thermal sensor initialized")
        return mlx
    except Exception as e:
        print(f"Failed to initialize MLX90640: {e}")
        return None

def read_thermal_value(mlx, frame, max_retries=5):
    """Read thermal frame, return 24x32 array or None on failure."""
    retry_count = 0
    while retry_count < max_retries:
        try:
            mlx.getFrame(frame)
            return np.reshape(frame, (24, 32))
        except (ValueError, RuntimeError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"[Thermal Error] Failed after {max_retries} retries: {e}")
                return None
            time.sleep(0.1)
    return None

# ------------------ PLOT SETUP ------------------ #

def setup_plot():
    """Set up dual subplots for radar and thermal."""
    global fig, radar_ax, thermal_ax
    fig = plt.figure(figsize=(12, 5))

    # Radar subplot
    radar_ax = fig.add_subplot(121)
    radar_line, = radar_ax.plot([], [], drawstyle='steps-post', linewidth=2, color='b')
    radar_ax.set_ylim(-0.2, 1.2)
    radar_ax.set_yticks([0, 1])
    radar_ax.set_yticklabels(["No Presence", "Presence"])
    radar_ax.set_xlabel("Time (s)")
    radar_ax.set_ylabel("Presence")
    radar_ax.set_title("Radar Presence Detection")
    radar_ax.grid(True)

    # Thermal subplot
    thermal_ax = fig.add_subplot(122)
    thermal_data = np.zeros((24, 32))
    thermal_img = thermal_ax.imshow(thermal_data, vmin=0, vmax=60, cmap='inferno', interpolation='bilinear')
    cbar = fig.colorbar(thermal_img, ax=thermal_ax)
    cbar.set_label('Temperature [C]', fontsize=14)
    thermal_ax.set_title("Thermal Image")

    return radar_line, thermal_img

# ------------------ UPDATE FUNCTION ------------------ #

def update_plot(frame, ser, mlx, radar_line, thermal_img, state, thermal_frame):
    """Update radar and thermal plots."""
    now = time.time()
    radar_presence = read_presence_value(ser)
    
    if radar_presence is None:
        #print("[Debug] Skipping update due to invalid radar data")
        return [radar_line, thermal_img]

    # Warmup period: buffer radar readings
    if now - state['start_time'] < state['warmup_time']:
        #print(f"[Debug] Warming up, buffer: {state['radar_buffer']}")
        state['radar_buffer'].append(radar_presence)
        if len(state['radar_buffer']) >= 5 and sum(state['radar_buffer'][-5:]) >= 4:  # Stricter 4/5
            state['radar_locked'] = True
            state['radar_stable_count'] = state['radar_threshold']
            #print("[Debug] Early radar detection during warmup")
        return [radar_line, thermal_img]

    # Radar detection logic
    new_detection = False
    if radar_presence == 1:
        state['radar_stable_count'] += 1
        if not state['radar_locked'] and state['radar_stable_count'] >= state['radar_threshold']:
            state['radar_locked'] = True
            new_detection = True
            print("Human detected entering active state")
    else:
        state['radar_stable_count'] = 0
        if state['radar_locked'] and state['radar_stable_count'] >= state['radar_threshold']:
            state['radar_locked'] = False
            print("Radar: No human detected")

    # Only update plots and read thermal if radar detects human
    if state['radar_locked'] or radar_presence == 1:
        # Radar plot
        current_time = now - state['start_time']
        timestamps.append(current_time)
        presence_data.append(radar_presence)
        max_points = 100
        timestamps_trimmed = timestamps[-max_points:]
        presence_trimmed = presence_data[-max_points:]
        radar_line.set_data(timestamps_trimmed, presence_trimmed)
        radar_ax.relim()
        radar_ax.autoscale_view()

        # Read thermal data
        thermal_data = read_thermal_value(mlx, thermal_frame)
        if thermal_data is not None:
            thermal_img.set_data(np.fliplr(thermal_data))
            thermal_img.set_clim(vmin=np.min(thermal_data), vmax=np.max(thermal_data))
            max_temp = np.max(thermal_data)
            print(f"Sensor Data: Radar={radar_presence}, Thermal Max Temp={max_temp:.1f}C")
        else:
            print("[Debug] Thermal data unavailable")
            thermal_data = np.zeros((24, 32))  # Fallback for image saving

        # Save images and send email on new radar detection
        if new_detection and now - state['last_alert_time'] > state['email_cooldown']:
            timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            radar_filename = os.path.join(OUTPUT_DIR, f"radar_plot_{timestamp}.png")
            thermal_filename = os.path.join(OUTPUT_DIR, f"thermal_{timestamp}_upscaled.png")

            # Save radar plot
            fig.savefig(radar_filename, bbox_inches='tight')
            print(f"Saved radar plot: {radar_filename}")

            # Save thermal image
            norm_array = (thermal_data - np.min(thermal_data)) / (np.max(thermal_data) - np.min(thermal_data)) * 255
            norm_array = norm_array.astype(np.uint8)
            colored_array = cv2.applyColorMap(np.fliplr(norm_array), cv2.COLORMAP_INFERNO)
            upscaled_array = cv2.resize(colored_array, (32 * UPSCALE_FACTOR, 24 * UPSCALE_FACTOR), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(thermal_filename, upscaled_array)
            print(f"Saved thermal image: {thermal_filename}")

            # Send email
            send_email_alert(thermal_filename, radar_filename)
            state['last_alert_time'] = now

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    return [radar_line, thermal_img]


# ------------------ START PLOTTING ------------------ #

def start_plotting(ser, mlx):
    """Set up dual plots for radar and thermal data."""
    radar_line, thermal_img = setup_plot()

    # State dictionary
    state = {
        'start_time': time.time(),
        'warmup_time': 5,
        'radar_locked': False,
        'radar_stable_count': 0,
        'radar_threshold': RADAR_THRESHOLD,
        'last_alert_time': 0,
        'email_cooldown': EMAIL_COOLDOWN,
        'radar_buffer': []
    }

    # Thermal frame buffer
    thermal_frame = np.zeros((24 * 32,))

    ani = FuncAnimation(fig, update_plot, fargs=(ser, mlx, radar_line, thermal_img, state, thermal_frame),
                       interval=100, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

# ------------------ MAIN EXECUTION ------------------ #

if __name__ == "__main__":
    ser = None
    mlx = None
    try:
        print("Initializing radar sensor...")
        ser = setup_serial()
        if ser is None:
            raise RuntimeError("Radar initialization failed")

        print("Initializing thermal sensor...")
        mlx = setup_thermal()
        if mlx is None:
            raise RuntimeError("Thermal initialization failed")

        print("Starting dual sensor plotting...")
        start_plotting(ser, mlx)
    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Fatal Error {e}")
    finally:
        if 'ser' in locals() and ser is not None:
            try:
                ser.close()
                print("Serial port closed")
            except KeyboardInterrupt:
               print("Program terminated by user") 
            except Exception as e:
                print(f"Failed to close serial port: {e}")
        plt.close('all')
        print("Plot windows closed")