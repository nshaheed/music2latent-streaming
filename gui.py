import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from music2latent.audio import extract_envelope_np

class InteractiveGraph(FigureCanvas):
    def __init__(self, parent=None, width=5, height=8, dpi=100):
        # Initialize the Matplotlib Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        # Data initialization (100 points)
        self.x = np.linspace(0, 10, 100)
        self.top_data = np.sin(self.x)  # Static-ish data (-1 to 1)
        self.bottom_data = np.zeros(100) # Interactive data (0 to 1)

        # Setup Subplots
        self.ax_top = self.fig.add_subplot(211)
        self.ax_bottom = self.fig.add_subplot(212)
        
        # Configure Top Plot
        self.ax_top.set_ylim(-1, 1)
        self.ax_top.set_title("Top Graph (-1 to 1)")
        self.top_line, = self.ax_top.plot(self.x, self.top_data, color='blue')

        # Configure Bottom Plot
        self.ax_bottom.set_ylim(0, 1)
        self.ax_bottom.set_title("Draw Here (0 to 1)")
        self.bottom_line, = self.ax_bottom.plot(self.x, self.bottom_data, color='red', lw=2)

        self.fig.tight_layout()

        # Interactivity State
        self.drawing = False
        
        # Connect Matplotlib Events
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax_bottom: return
        self.drawing = True
        self.update_data(event.xdata, event.ydata)

    def on_release(self, event):
        self.drawing = False
        print("Updated Bottom Array:", self.bottom_data) # Exposed array

    def on_motion(self, event):
        if not self.drawing or event.inaxes != self.ax_bottom: return
        self.update_data(event.xdata, event.ydata)

    def update_data(self, x_pos, y_pos):
        # Find the index in the numpy array closest to the mouse X position
        idx = (np.abs(self.x - x_pos)).argmin()
        
        # Clamp Y value between 0 and 1
        val = np.clip(y_pos, 0, 1)
        
        # Update the specific point in the array
        self.bottom_data[idx] = val
        self.bottom_line.set_ydata(self.bottom_data)
        
        # Redraw the canvas
        self.draw()

    def get_bottom_array(self):
        """Method to expose the manipulated array."""
        return self.bottom_data

    def update_top_plot(self, x, y):
        """Update top plot"""
        print("update_top_plot")
        self.x = x
        self.top_data = y
        self.ax_top.clear()
        self.ax_top.set_ylim(-1, 1)
        self.top_line, = self.ax_top.plot(x, y, color='blue')
        return self.top_line

    def update_bottom_plot(self, x, y):
        """Update bottom plot"""
        print("update_top_plot")
        self.bottom_data = y
        self.ax_bottom.clear()
        self.ax_bottom.set_ylim(0, 1)
        self.bottom_line, = self.ax_bottom.plot(x, y, color='red')
        return self.bottom_line

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # breakpoint()
        super().__init__()
        self.setWindowTitle("Interactive Stacked Graphs")

        # Main Layout
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)
        layout = QtWidgets.QVBoxLayout(self.main_widget)

        # Add the Graph
        self.canvas = InteractiveGraph(self, width=6, height=8)
        layout.addWidget(self.canvas)

        # Helper label
        self.label = QtWidgets.QLabel("Click and drag on the bottom graph to draw.")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        import soundfile as sf
        import math

        hop = 128*4 # hop size
                
        file = "Kick 909 2.wav"
        wv, sr = sf.read(file, dtype="float32", always_2d=True)

        # fill in end with zeroes to make hop-size divisible
        env_len = math.ceil(wv.shape[0] / hop)
        wv = wv[:,0]
        wv = np.pad(wv, (0, env_len*hop - wv.shape[0]), 'constant', constant_values=(0,0))
        wv = np.expand_dims(wv, 0)

        _, _, data_env = extract_envelope_np(wv, target_length=env_len)
        # reshape to add a channel dim
        # data_env = data_env.unsqueeze(1)

        x = np.arange(0, data_env.shape[1])
        self.canvas.update_bottom_plot(x, data_env[0])

        x = np.arange(0, wv.shape[1])
        self.canvas.update_top_plot(x, wv[0])

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
