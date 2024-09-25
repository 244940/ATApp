import tkinter as tk
from face_scanner_app import FaceScannerApp

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceScannerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()