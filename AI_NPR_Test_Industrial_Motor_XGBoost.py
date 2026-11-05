# ---------------- IMPORT ----------------
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import joblib
import threading
import serial
import serial.tools.list_ports
import time
from tkinter import messagebox
import csv
import os
from datetime import datetime
import yagmail
import pandas as pd   # ⭐ FIX

# ---------------- SETTINGS ----------------
EMAIL_USER = "your_email@gmail.com"
EMAIL_PASS = "your_app_password"
EMAIL_TO = ["receiver@gmail.com"]

CSV_FILE = "motor_log.csv"

# ---------------- LOAD MODEL ----------------
MODEL_FILE = "motor_fault_model.pkl"
model = joblib.load(MODEL_FILE)

# ---------------- CREATE CSV ----------------
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Time", "Voltage", "Current", "Temperature", "Status", "FaultType"])

# ---------------- EMAIL FUNCTION ----------------
def send_email(subject, body):
    def task():
        try:
            yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)
            yag.send(EMAIL_TO, subject, body)
        except Exception as e:
            print("Email error:", e)
    threading.Thread(target=task, daemon=True).start()

# ---------------- UI SETUP ----------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("1500x900")
app.title("AI Motor Monitoring System")

ser = None
auto_mode = True
running = True

GREEN = "#16A34A"
RED = "#DC2626"

last_email_time = 0

# ---------------- TITLE ----------------
ctk.CTkLabel(app, text="⚡ AI MOTOR MONITORING", font=("Arial", 26, "bold")).pack(pady=10)

# ---------------- COM PORT ----------------
port_frame = ctk.CTkFrame(app)
port_frame.pack()

ports = [p.device for p in serial.tools.list_ports.comports()]
port_var = ctk.StringVar(value=ports[0] if ports else "")

ctk.CTkLabel(port_frame, text="COM Port:").pack(side="left")
ctk.CTkOptionMenu(port_frame, values=ports, variable=port_var).pack(side="left")

def connect_serial():
    global ser
    try:
        ser = serial.Serial(port_var.get(), 9600, timeout=1)
        time.sleep(1)
        messagebox.showinfo("Connected", port_var.get())
    except Exception as e:
        messagebox.showerror("Error", str(e))

ctk.CTkButton(port_frame, text="Connect", command=connect_serial).pack(side="left")

# ---------------- CARDS ----------------
top = ctk.CTkFrame(app)
top.pack(fill="x")

def card(parent, name):
    f = ctk.CTkFrame(parent)
    f.pack(side="left", expand=True, fill="both", padx=10, pady=10)
    ctk.CTkLabel(f, text=name).pack()
    val = ctk.CTkLabel(f, text="--", font=("Arial", 20))
    val.pack()
    return val

v_lbl = card(top, "Voltage (V)")
i_lbl = card(top, "Current (A)")
t_lbl = card(top, "Temperature (°C)")

# ---------------- STATUS ----------------
status_lbl = ctk.CTkLabel(app, text="Status", font=("Arial", 18))
status_lbl.pack()

fault_lbl = ctk.CTkLabel(app, text="Fault: None", font=("Arial", 16))
fault_lbl.pack()

fault_type_lbl = ctk.CTkLabel(app, text="Fault Type: None", font=("Arial", 16))
fault_type_lbl.pack()

# ---------------- CONTROL ----------------
ctrl = ctk.CTkFrame(app)
ctrl.pack(pady=5)

def auto():
    global auto_mode
    auto_mode = True

def manual():
    global auto_mode
    auto_mode = False

def relay_on():
    if ser: ser.write(b'1')

def relay_off():
    if ser: ser.write(b'2')

ctk.CTkButton(ctrl, text="AUTO", command=auto).pack(side="left", padx=5)
ctk.CTkButton(ctrl, text="MANUAL", command=manual).pack(side="left", padx=5)
ctk.CTkButton(ctrl, text="Relay ON", command=relay_on).pack(side="left", padx=5)
ctk.CTkButton(ctrl, text="Relay OFF", command=relay_off).pack(side="left", padx=5)

# ---------------- MAIN FRAME ----------------
main = ctk.CTkFrame(app)
main.pack(fill="both", expand=True)

# -------- GRAPH --------
graph_frame = ctk.CTkFrame(main)
graph_frame.pack(side="left", fill="both", expand=True)

MAX_LEN = 50
v_hist, i_hist, t_hist = deque(maxlen=MAX_LEN), deque(maxlen=MAX_LEN), deque(maxlen=MAX_LEN)

fig, ax = plt.subplots(figsize=(6,4))
line_v, = ax.plot([], [], label="Voltage")
line_i, = ax.plot([], [], label="Current")
line_t, = ax.plot([], [], label="Temperature")
ax.legend()
ax.grid()

canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True)

def update_graph():
    line_v.set_data(range(len(v_hist)), list(v_hist))
    line_i.set_data(range(len(i_hist)), list(i_hist))
    line_t.set_data(range(len(t_hist)), list(t_hist))
    ax.set_xlim(0, MAX_LEN)

    all_vals = list(v_hist)+list(i_hist)+list(t_hist)
    if all_vals:
        ax.set_ylim(min(all_vals)-5, max(all_vals)+5)

    canvas.draw_idle()

# -------- LOG UI --------
log_frame = ctk.CTkFrame(main)
log_frame.pack(side="right", fill="both", expand=True)

ctk.CTkLabel(log_frame, text="📊 Data Log", font=("Arial",16,"bold")).pack()

data_box = ctk.CTkTextbox(log_frame)
data_box.pack(fill="both", expand=True)

data_box.insert("end", "Time | Voltage | Current | Temp | Status | Type\n")
data_box.insert("end", "-"*70 + "\n")

# ---------------- LOOP ----------------
def update():
    global last_email_time, ser

    try:
        if ser is None:
            app.after(200, update)
            return

        line = ser.readline().decode().strip()

        if not line:
            app.after(200, update)
            return

        print("RAW:", line)

        p = line.split(",")
        if len(p) < 3:
            app.after(200, update)
            return

        v = float(p[0])
        i = float(p[1])
        t = float(p[2])

        # -------- AI PREDICTION --------
        input_data = pd.DataFrame([[v, i, t]],
                                  columns=["Voltage", "Current", "Temperature"])
        pred = model.predict(input_data)[0]

        # -------- FAULT TYPE LOGIC --------
        if pred == 1:
            status = "FAULT"

            if v >= 250 or v <= 150:
                fault_type = "Voltage Fault"
            elif t >= 40:
                fault_type = "Temperature Fault"
            elif i >= 15:
                fault_type = "Current Fault"
            else:
                fault_type = "Unknown Fault"

            fault = "Motor Fault"
            relay_cmd = b'2'
        else:
            status = "NORMAL"
            fault = "Normal"
            fault_type = "No Fault"
            relay_cmd = b'1'

        # -------- UI UPDATE --------
        v_lbl.configure(text=f"{v:.1f} V")
        i_lbl.configure(text=f"{i:.2f} A")
        t_lbl.configure(text=f"{t:.1f} °C")

        fault_lbl.configure(text=f"Fault: {fault}")
        fault_type_lbl.configure(text=f"Type: {fault_type}")

        if status == "FAULT":
            status_lbl.configure(text="🚨 FAULT", text_color=RED)

            if time.time() - last_email_time > 30:
                send_email("Motor Fault Alert",
                           f"{fault_type}\nV:{v} I:{i} T:{t}")
                last_email_time = time.time()
        else:
            status_lbl.configure(text="✅ NORMAL", text_color=GREEN)

        # -------- AUTO CONTROL --------
        if auto_mode and ser:
            ser.write(relay_cmd)

        # -------- CSV LOG --------
        now = datetime.now()
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                v, i, t, status, fault_type
            ])

        # -------- LOG UI --------
        data_box.insert("end",
            f"{now.strftime('%H:%M:%S')} | {v:.1f} | {i:.2f} | {t:.1f} | {status} | {fault_type}\n")
        data_box.see("end")

        # -------- GRAPH --------
        v_hist.append(v)
        i_hist.append(i)
        t_hist.append(t)
        update_graph()

    except Exception as e:
        print("ERROR:", e)

    app.after(200, update)

# ---------------- EXIT ----------------
def exit_app():
    global running
    running = False
    app.destroy()

ctk.CTkButton(app, text="Exit", command=exit_app).pack(pady=10)

app.after(200, update)
app.mainloop()