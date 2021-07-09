import tkinter as tk

root = tk.Tk()

switch_frame = tk.Frame(root)
switch_frame.pack()

switch_variable = tk.StringVar(value="off")
off_button = tk.Radiobutton(switch_frame, text="Off", variable=switch_variable,
                            indicatoron=False, value="off", width=8)
low_button = tk.Radiobutton(switch_frame, text="Low", variable=switch_variable,
                            indicatoron=False, value="low", width=8)
med_button = tk.Radiobutton(switch_frame, text="Medium", variable=switch_variable,
                            indicatoron=False, value="medium", width=8)
high_button = tk.Radiobutton(switch_frame, text="High", variable=switch_variable,
                             indicatoron=False, value="high", width=8)
off_button.pack(side="left")
low_button.pack(side="left")
med_button.pack(side="left")
high_button.pack(side="left")

root.mainloop()