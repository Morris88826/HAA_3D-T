import tkinter as tk
import os
from PIL import ImageTk,Image 
import os
import json
import numpy as np
from skeleton import joints_key_2_index, joints_index_2_key
from temp_page import Temporal_window
from copy import deepcopy

class Page2(tk.Frame):
    def __init__(self, root, controller, parent=None):
        tk.Frame.__init__(self, root)

        self.save_v1 = True

        self.root = root
        self.controller = controller
        self.parent = parent

        self.raw_images_root = './dataset/raw' 
        self.w_height = self.controller.winfo_screenheight()
        self.w_width = self.controller.winfo_screenwidth()

        # Page 2 interface
        self.ratio = 3/4
        self.left_frame = tk.Frame(self, width=self.w_width*self.ratio, height=self.w_height*self.ratio)
        self.left_frame.pack_propagate(0)
        self.left_frame.pack(side="left", fill="both", expand=True)
        
        self.right_frame = tk.Frame(self, width =self.w_width*(1-self.ratio), height=700)
        self.right_frame.pack_propagate(0)
        self.right_frame.pack(side="right", fill="both", expand=False)

        self.label = tk.Label(self.right_frame, text="")
        self.label.pack(side="top")

        self.current_joint_label = tk.Label(self.right_frame, text="")
        self.current_joint_label.pack(fill="both", padx=5, pady=5)


        reset_button = tk.Button(self.right_frame, text="Reset Zoom", command=self.reset_zoom)
        reset_button.pack(fill="both", padx=5, pady=5)

        load_alphapose_button = tk.Button(self.right_frame, text="Load from alphapose", command=self.load_from_alphapose)
        load_alphapose_button.pack(fill="both", padx=5, pady=5)

        load_joints2d_button = tk.Button(self.right_frame, text="Load from joints2d", command=self.load_from_joints2d)
        load_joints2d_button.pack(fill="both", padx=5, pady=5)

        label_everything_button = tk.Button(self.right_frame, text="Label Everything or Stop(L)", command=self.label_everything)
        label_everything_button.pack(fill="both", padx=5, pady=5)

        info_label = tk.Label(self.right_frame, text="To label one joint: (4,5), (q-u), (a-j),(v)", font=("Helvetica", 8))
        info_label.pack(fill="both")
        info_label2 = tk.Label(self.right_frame, text="Press (1) to change a joint to hidden. Press (2) to delete joint.", font=("Helvetica", 8))
        info_label2.pack(fill="both")

        self.instruction = tk.Canvas(self.right_frame, width=self.w_width*(1-self.ratio)*(3/5), height=self.w_width*(1-self.ratio)*(3/5))
        self.instruction.pack(side="top")
        img = Image.open("./instruction.jpg")
        img = img.resize((int(self.w_width*(1-self.ratio)*(3/5)), int(self.w_width*(1-self.ratio)*(3/5))))
        self.inst_img = ImageTk.PhotoImage(img)
        self.instruction.create_image(0,0, image=self.inst_img, anchor="nw")
        self.instruction_position = {
            0: [87/192, 98/192],
            1: [78/192, 107/192],
            2: [73/192, 144/192],
            3: [72/192, 179/192],
            4: [101/192, 107/192],
            5: [106/192, 144/192],
            6: [107/192, 179/192],
            7: [87/192, 77/192],
            8: [87/192, 54/192],
            9: [87/192, 41/192],
            10: [87/192, 23/192],
            11: [111/192, 50/192],
            12: [120/192, 76/192],
            13: [123/192, 109/192],
            14: [68/192, 50/192],
            15: [59/192, 76/192],
            16: [56/192, 109/192]
        }

        self.instruction_size = self.w_width*(1-self.ratio)*(3/5)

        all_shown_button = tk.Button(self.right_frame, text="Mark all unoccluded", command=self.shown_all)
        all_shown_button.pack(fill="both", padx=5, pady=5)

        auto_interpolate_button = tk.Button(self.right_frame, text="Auto interpolate", command=self.auto_interpolate)
        auto_interpolate_button.pack(fill="both", padx=5, pady=5)

        smoothing_button = tk.Button(self.right_frame, text="Motion Smoothing", command=self.gaussian_smoothing)
        smoothing_button.pack(fill="both", padx=5, pady=5)

        temporal_prediction_button = tk.Button(self.right_frame, text="Temporal Prediction", command=self.temporal_predicting)
        temporal_prediction_button.pack(fill="both", padx=5, pady=5)

        self.bone_label = tk.Label(self.right_frame, text="Set line width of Bone", fg="black")
        self.bone_label.pack(fill="both")
        self.bone_scale = tk.Scale(self.right_frame, from_=1, to=10, orient=tk.HORIZONTAL, command=self.changed_bone)
        self.bone_scale.pack(side="top", fill="both")
        self.bone_scale.set(2)

        self.joint_circle_label = tk.Label(self.right_frame, text="Set circle size of joint", fg="black")
        self.joint_circle_label.pack(fill="both")
        self.joint_circle_scale = tk.Scale(self.right_frame, from_=1, to=10, orient=tk.HORIZONTAL, command=self.changed_joint_circle)
        self.joint_circle_scale.pack(side="top", fill="both")
        self.joint_circle_scale.set(2)

        save_exit_button = tk.Button(self.right_frame, text="Save and Exit", command=self.save_and_exit)
        save_exit_button.pack(fill="both", padx=5, pady=5, side="bottom")

        exit_button = tk.Button(self.right_frame, text="Exit without Saving", command=self.exit)
        exit_button.pack(fill="both", padx=5, pady=5, side="bottom")

        save_button = tk.Button(self.right_frame, text="Save to joints2d", command=self.save)
        save_button.pack(fill="both", padx=5, pady=5, side="bottom")

        self.alert_label = tk.Label(self.right_frame, text="", fg="red", font=("Helvetica", 8))
        self.alert_label.pack(fill="both", padx=5, pady=5)


        self.canvas = tk.Canvas(self.left_frame, width=self.w_width*self.ratio, height=self.w_height*self.ratio)
        self.canvas.pack(side="top", padx=10, pady=10, anchor="nw", fill="both", expand=True)

        lb_frame = tk.Frame(self.left_frame, width=self.w_width*self.ratio, height=self.w_height*(1-self.ratio))
        lb_frame.pack(side="top", fill="x", anchor="nw", expand=False, padx=20, pady=20)

        self.scale = tk.Scale(lb_frame, from_=0, to=1, orient=tk.HORIZONTAL, command=self.changed_frame)
        self.scale.pack(side="top", fill="both")

        # Page 2 information
        self.current_frame = 1
        self.current_joint = -1
        self.video_length = -1
        self.image_corner = [self.w_width*self.ratio*(0.1), self.w_height*self.ratio*(0.1)]
        self.zoom = -4
        self.position = [-1, -1]
        self.joint_circle_size = 2
        self.bone_width = 3
        self.label_everything_mode = False
        self.joint2d_init()
        self.user_selected = {}

        self.bones_indices = [
            [[0,4],[4,5],[5,6],[8,11],[11,12],[12,13]], # left -> pink
            [[0,1],[1,2],[2,3],[8,14],[14,15],[15,16]], # right -> blue
            [[0,7],[7,8],[8,9],[9,10]] # black
        ] # left-> pink, right->blue
        self.bone_colors = ['pink','blue','gray']

        self.key_bind_table = ['v','g','h','j','d','s','a','f','r','4','5','e','w','q','t','y','u']
        self.label_everything_mode = False

    def initialize(self):
        self.key_binding()
        self.path = self.raw_images_root + '/{}/{}'.format(self.parent.current_video[0], self.parent.current_video[1])
        self.video_length = len(os.listdir(self.path))
        self.current_frame = 1
        self.current_joint = -1
        self.image_corner = [self.w_width*self.ratio*(0.1), self.w_height*self.ratio*(0.1)]
        self.zoom = -4
        self.position = [-1, -1]
        self.joint_circle_size = 2
        self.canvas.delete("oval")
        self.canvas.delete("bone")
        self.label.config(text="Current frame: {}".format(self.current_frame))
        self.current_joint_label.config(text="Current joint: None")
        self.scale.configure(from_=1, to=self.video_length)
        self.scale.set(1)
        self.joint2d_init()

        self.user_selected = {}
        for i in range(17):
            self.user_selected[i] = {}

    def key_binding(self):
        self.canvas.bind("<Button-1>", self.event_handler)
        self.canvas.bind("<B1-Motion>", self.event_handler)
        self.canvas.bind("<ButtonRelease-1>", self.event_handler)
        self.canvas.bind("<MouseWheel>", self.event_handler)

        self.controller.bind('z', self.event_handler)
        self.controller.bind('x', self.event_handler)
        self.controller.bind('1', self.event_handler)
        self.controller.bind('2', self.event_handler)
        self.controller.bind('3', self.event_handler)
        self.controller.bind('l', self.event_handler)
        for key in self.key_bind_table:
            self.controller.bind(key, self.event_handler)

    
    def key_unbinding(self):
        self.controller.unbind('z')
        self.controller.unbind('x')
        self.controller.unbind('1')
        self.controller.unbind('2')
        self.controller.unbind('3')
        self.controller.unbind('l')
        for key in self.key_bind_table:
            self.controller.unbind(key)


    def load_from_alphapose(self):
        pass
    
    def joint2d_init(self):
        self.joints2d = {}
        for i in range(self.video_length):
            frame_id = i + 1
            tmp = {}
            for j in range(len(joints_index_2_key.keys())):
                tmp[j] = None
            self.joints2d[frame_id] = tmp
    
    def load_from_joints2d(self):
        path = self.path.replace('raw', 'joints2d')
        if not os.path.exists(path):
            return
        
        for i in range(self.video_length):
            frame_id = i + 1
            with open(path+'/{:04d}.json'.format(frame_id)) as jsonfile:
                d = json.load(jsonfile)
            d = np.array(d).reshape((17, -1))
            if d.shape[-1] == 2:
                tmp = d
                d = np.ones((17,3))
                d[:,:2] = tmp

            for j in range(len(joints_index_2_key.keys())):
                self.joints2d[frame_id][j] = list(d[j])
        
        self.update_skeleton()
    
    def auto_interpolate(self):
        for joint in self.user_selected:
            start_frame = 1
            for idx, frame in enumerate(sorted(self.user_selected[joint])):
                if idx == 0:
                    for f in range(start_frame, frame):
                        self.joints2d[f][joint] = deepcopy(self.user_selected[joint][frame])
                        self.joints2d[f][joint][-1] = 1
                else:
                    dif_x = self.user_selected[joint][frame][0] - self.user_selected[joint][start_frame][0]
                    dif_y = self.user_selected[joint][frame][1] - self.user_selected[joint][start_frame][1]
                    for f in range(start_frame+1, frame):
                        delta = (f-start_frame)/(frame - start_frame)
                        i_x = self.user_selected[joint][start_frame][0] + dif_x*delta
                        i_y = self.user_selected[joint][start_frame][1] + dif_y*delta
                        self.joints2d[f][joint] = [i_x, i_y, 1]
                
                if idx == (len(self.user_selected[joint])-1):
                    for f in range(frame+1, self.video_length+1):
                        self.joints2d[f][joint] = deepcopy(self.user_selected[joint][frame])
                        self.joints2d[f][joint][-1] = 1
                start_frame = frame


    def label_everything(self):
        if self.label_everything_mode == True:
            self.label_everything_mode = False
            self.instruction.delete("oval")
            self.current_joint = -1
            self.current_joint_label.config(text="Current joint: None")
            return 
        self.label_everything_mode = True
        self.current_joint = 0
        self.instruction.delete("oval")
        _x, _y = list(np.array(self.instruction_position[self.current_joint]) * self.instruction_size)
        self.instruction.create_oval(_x-3, _y-3, _x+3, _y+3, fill='red', outline='red', tags="oval")
        self.current_joint_label.config(text="Current joint: {}".format(joints_index_2_key[self.current_joint]))

    def gaussian_smoothing(self):
        pass

    def temporal_predicting(self):
        self.new = tk.Toplevel(self.root)
        self.temporal_window = Temporal_window(self.new, self)

    def shown_all(self):
        for joint in self.joints2d[self.current_frame]:
            if self.joints2d[self.current_frame][joint] is not None:
                self.joints2d[self.current_frame][joint][-1] = 1
        self.update_skeleton()

    def changed_bone(self, value):
        self.bone_width = int(value)
        self.update_skeleton()

    def changed_joint_circle(self, value):
        self.joint_circle_size = int(value)
        self.update_skeleton()
    

    def reset_zoom(self):
        self.zoom = -4
        self.image_corner = [self.w_width*self.ratio*(0.1), self.w_height*self.ratio*(0.1)]
        self.bone_width = 2
        self.joint_circle_size = 2
        self.bone_scale.set(self.bone_width)
        self.joint_circle_scale.set(self.joint_circle_size)
        self.show()

    def changed_frame(self, value):
        self.current_frame = int(value)
        self.label.config(text="Current frame: {}".format(self.current_frame))
        self.show()

    def event_handler(self, event):
        if str(event.type) == "MouseWheel":
            if event.delta < 0:
                self.zoom += 1
                self.zoom = min(self.zoom, 10)
            else:
                self.zoom -= 1
                self.zoom = max(self.zoom, -10)
            self.show()
        
        elif str(event.type) == "ButtonPress":
            if self.current_joint == -1:
                self.position = [event.x, event.y]
            else:
                scaler = 2**(self.zoom/5)
                position = [(event.x-self.image_corner[0])/scaler, (event.y-self.image_corner[1])/scaler]

                if self.joints2d[self.current_frame][self.current_joint] is None:
                    self.joints2d[self.current_frame][self.current_joint] = [position[0], position[1], 1]
                else:
                    self.joints2d[self.current_frame][self.current_joint][:2] = [position[0], position[1]]
                
                self.user_selected[self.current_joint][self.current_frame] = [position[0], position[1], 1]
                
            
                if self.label_everything_mode:
                    self.current_joint += 1
                    self.instruction.delete("oval")
                    if self.current_joint == 17:
                        self.current_joint = -1
                        self.label_everything_mode = False
                self.update_skeleton()
        elif str(event.type) == "Motion":
            if self.current_joint == -1:
                move = [event.x-self.position[0], event.y-self.position[1]]
                self.position = [event.x, event.y]
                self.image_corner[0] += move[0]
                self.image_corner[1] += move[1]
                self.show()

        elif str(event.type) == "KeyPress":
            if event.char == 'z':
                self.current_frame -= 1
                self.current_frame = max(1, self.current_frame)
                self.changed_frame(self.current_frame)
                self.scale.set(self.current_frame)
            elif event.char == 'x':
                self.current_frame += 1
                self.current_frame = min(self.video_length, self.current_frame)
                self.changed_frame(self.current_frame)
                self.scale.set(self.current_frame)
            
            elif event.char == '1' or event.char == '2' or event.char=='3':
                if self.current_joint != -1:
                    if self.joints2d[self.current_frame][self.current_joint] != None:
                        if event.char == '1':
                            if not self.label_everything_mode:
                                self.joints2d[self.current_frame][self.current_joint][-1] = -1*self.joints2d[self.current_frame][self.current_joint][-1]
                            else:
                                self.joints2d[self.current_frame][self.current_joint][-1] = -1
                                self.current_joint += 1
                                if self.current_joint == 17:
                                    self.current_joint = -1
                        elif event.char == '3':
                            if self.label_everything_mode:
                                self.joints2d[self.current_frame][self.current_joint][-1] = 1
                                self.current_joint += 1
                                if self.current_joint == 17:
                                    self.current_joint = -1
                            


                        else:
                            self.joints2d[self.current_frame][self.current_joint] = None
                            try:
                                self.user_selected[self.current_joint].pop(self.current_frame)
                            except:
                                pass

                        self.update_skeleton()
            elif event.char == 'l':
                self.label_everything()

            elif event.char in self.key_bind_table:
                self.label_everything_mode = False
                self.instruction.delete("oval")
                index = self.key_bind_table.index(event.char)
                if self.current_joint == index:
                    self.current_joint = -1
                else:
                    self.current_joint = index
                self.update_skeleton()

    def show(self):
        path = self.path + '/{:04d}.png'.format(self.current_frame)
        self.image = Image.open(path)
        self.img_size = self.image.size
        self.update_image()
        self.update_skeleton()

    def update_skeleton(self):
        self.instruction.delete("oval")

        if self.current_joint == -1:
            self.current_joint_label.config(text="Current joint: None")
        else:
            _x, _y = list(np.array(self.instruction_position[self.current_joint]) * self.instruction_size)
            self.instruction.create_oval(_x-3, _y-3, _x+3, _y+3, fill='red', outline='red', tags="oval")
            self.current_joint_label.config(text="Current joint: {}".format(joints_index_2_key[self.current_joint]))

        try:
            skeleton = self.joints2d[self.current_frame]
        except:
            return

        self.canvas.delete("oval")
        self.canvas.delete("bone")
        
        scaler = 2**(self.zoom/5)
        # draw bones
        for i, bones in enumerate(self.bones_indices):
            for bone in (bones):
                start = bone[0]
                end = bone[1]
                if skeleton[start] is not None and skeleton[end] is not None:
                    x0, y0 = list(np.array(self.image_corner)+np.array(skeleton[start][:2])*scaler)
                    x1, y1 = list(np.array(self.image_corner)+np.array(skeleton[end][:2])*scaler)
                    self.canvas.create_line(x0,y0,x1,y1, fill=self.bone_colors[i], width=self.bone_width, tags='bone')

        # draw dots
        for key, value in zip(skeleton.keys(), skeleton.values()):
            if value is not None:
                r= self.joint_circle_size
                
                color = 'red' if key != self.current_joint else 'green'
                color = color if value[-1] == 1 else 'black'

                position = [self.image_corner[0]+value[0]*scaler, self.image_corner[1]+value[1]*scaler]
                self.canvas.create_oval(position[0]-r, position[1]-r, position[0]+r, position[1]+r, 
                fill=color, outline=color, tags='oval')
    
    def update_image(self):
        size = self.img_size
        new_size = (int(size[0] * 2**(self.zoom/5)), int(size[1] * 2**(self.zoom/5)))
        self.image = self.image.resize(new_size)
        self.img = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(self.image_corner[0],self.image_corner[1], image=self.img, anchor="nw")
    
    def save(self):
        need_save = True
        # check if all are not None
        for frame in self.joints2d:
            for joint in self.joints2d[frame]:
                if self.joints2d[frame][joint] is None:
                    print('Frame {}, joint {} was not labelled'.format(frame, joint))
                    need_save = False
        
        if need_save:
            out_dir = './result'
            video_class = self.path.split('/')[-2]
            video_name = self.path.split('/')[-1]
            if not os.path.exists('{}/{}'.format(out_dir, video_class)):
                os.mkdir('{}/{}'.format(out_dir, video_class))
            out_path = '{}/{}/{}'.format(out_dir, video_class, video_name)
            if not os.path.exists(out_path):
                os.mkdir(out_path)

            for frame in self.joints2d:
                my_list = []
                for joint in sorted(self.joints2d[frame]):
                    if self.save_v1:
                        my_list += self.joints2d[frame][joint][:2]
                    else:
                        my_list += self.joints2d[frame][joint]
                
                with open(out_path+'/{:04d}.json'.format(frame), 'w') as jsonfile:
                    json.dump(my_list, jsonfile)
        
    def exit(self):
        self.key_unbinding()
        self.controller.show_frame('Page1')

    def save_and_exit(self):
        self.save()
        self.exit()

