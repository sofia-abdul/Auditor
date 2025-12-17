
import serial
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
from email.message import EmailMessage
import smtplib
import os
from datetime import datetime


# ------------------ CONFIG ------------------ #

EMAIL_SENDER = 'ayunvuai@gmail.com'
EMAIL_PASSWORD = 'fejw blvu wkau alti'  # Consider using os.environ for security
EMAIL_RECEIVER = 'williejoseph299@gmail.com'

# Threshold for human detection (adjust as needed)
#SENSOR_DATA_THRESHOLD = 5 # Celsius, typical human body temperature

# Output directory for saved images
OUTPUT_DIR = "radar_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global lists to store time and presence data
timestamps = []
presence_data = []

#Minimun time between email alert (seconds)
EMAIL_COOLDOWN = 300 #5 minutes

# ------------------ EMAIL FUNCTION ------------------ #

def send_email_alert(image_path):
    msg = EmailMessage()
    msg['Subject'] = 'Human Detected - Radar Plot Alert'
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.set_content(f'A human has been detected. See the attached presence plot: {image_path}')

    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='png', filename=os.path.basename(image_path))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"Email sent with {image_path}")
    except Exception as e:
        print(f"[Email Error] {e}")



#------ SENSOR SETUP -----#
def setup_serial(port="/dev/ttyACM0", baudrate=115200):
    """Initialize and return the serial connection."""
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"Connected to {port} at {baudrate} baud")
        return ser
    except serial.SerialException as e:
        print(f"Serial connection error: {e}")
        return None

def read_presence_value(ser):
    """Read and clean a line from serial, returning 0 or 1 if valid."""
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line == "0" or line == "1":
            return int(line)
        else:
            #print(f"Ignored invalid data: {line}")
            return None
    except Exception as e:
        print(f"Serial read error: {e}")
        return None

def update_plot(frame, ser, line_plot, state):
    """Update the plot with new data from the serial port."""
    now = time.time()
    presence = read_presence_value(ser)
    
    if presence is None:
        return line_plot,

    # Warmup period check
    if now - state['start_time'] < state['warmup_time']:
        print("Warming up...")
        return line_plot,

    # Stable detection logic
    if presence == 1:
        state['stable_count'] += 1
        if not state['presence_locked'] and state['stable_count'] >= state['threshold']:
            state['presence_locked'] = True
            print("Human detected entering active state")
            # Save and email plot only on new detection, with cooldown
            if now - state['last_alert_time'] > state['email_cooldown']:
                timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                filename = os.path.join(OUTPUT_DIR, f"radar_plot_{timestamp}.png")
                plt.savefig(filename, bbox_inches='tight')
                print(f"Saved plot image: {filename}")
                send_email_alert(filename)
                state['last_alert_time'] = now
    else:
        state['stable_count'] = 0
        if state['presence_locked'] and state['stable_count'] >= state['threshold']:
            state['presence_locked'] = False
            print("No human detected, exiting active state")

  # Only plot/log data after warmup and if locked or presence == 1
    if state['presence_locked'] or presence == 1:
        current_time = now - state['start_time']
        timestamps.append(current_time)
        presence_data.append(presence)

        # Limit to latest 100 points for smooth display
        max_points = 100
        timestamps_trimmed = timestamps[-max_points:]
        presence_trimmed = presence_data[-max_points:]

        line_plot.set_data(timestamps_trimmed, presence_trimmed)
        ax.relim()
        ax.autoscale_view()

        print(f"Sensor Data: {presence}")

    return line_plot,

def start_plotting(ser):
    """Set up the matplotlib animation for real-time plotting."""
    global fig, ax
    fig, ax = plt.subplots()
    line_plot, = ax.plot([], [], drawstyle='steps-post', linewidth=2, color='b')
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No Presence", "Presence"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Presence")
    ax.set_title("Real-Time Radar Presence Detection")
    ax.grid(True)

    # State dictionary for detection logic
    state = {
        'start_time': time.time(),
        'warmup_time': 3,  # seconds
        'presence_locked': False,
        'stable_count': 0,
        'threshold': 2,
        'last_alert_time': 0,
        'email_cooldown': EMAIL_COOLDOWN
    }

    ani = FuncAnimation(fig, update_plot, fargs=(ser, line_plot, state), interval=100, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    ser = setup_serial()
    if ser:
        print("Starting real-time plot...")
        try:
            start_plotting(ser)
        except KeyboardInterrupt:
            print("Plotting stopped by user.")
        finally:
            ser.close()
            print("Serial port closed.")
    else:
        print("Failed to open serial port.")


            















































































































