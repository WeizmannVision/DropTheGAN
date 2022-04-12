from tkinter import *
from tkinter.colorchooser import askcolor
from tkinter import filedialog
from PIL import ImageTk,Image, ImageGrab, ImageDraw
import os
import io
import math
from io import BytesIO
from image import *
from image_generation import *


class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()
        # self.root.configure(bg='white')
        self.pen_button = Button(self.root, text='use_pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.color_button = Button(self.root, text='pen color', command=self.choose_color)
        self.color_button.grid(row=0, column=1)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=VERTICAL)
        self.choose_size_button.grid(row=0, column=3)
        self.size_label = Label(self.root, text=' pen size:', font=('arial', 10, 'bold'))
        self.size_label.grid(row=0, column=2)



        self.open_button = Button(self.root, text='open_file', command=self.open_img)
        self.open_button.grid(row=1, column=0)


        self.eraser_button = Button(self.root, text='erase_all', command=self.use_eraser)
        self.eraser_button.grid(row=1, column=2)




        self.save_button = Button(self.root, text='save_file', command=self.save_as_png)
        self.save_button.grid(row=1, column=1)


        self.rect_button = Button(self.root, text='noise_region', command=self.use_rect)
        self.rect_button.grid(row=1, column=3)
        self.rect = None

        self.scale_label = Label(self.root, text=' pyramid depth: ', font=('arial', 10, 'bold'))
        self.choose_scale_button = Scale(self.root,  from_=0, to=9, orient=VERTICAL)
        self.choose_scale_button.grid(row=0, column=7)
        self.scale_label.grid(row=0, column=6)

        self.noise_var = StringVar(value='0.00')
        self.noise_label = Label(self.root, text=' noise variance: ', font=('arial', 10, 'bold'))
        self.choose_noise_button = Entry(self.root, textvariable=self.noise_var, width=4)
        self.choose_noise_button.grid(row=1, column=7)
        self.noise_label.grid(row=1, column=6)


        self.generate_button = Button(self.root, text='Generate!', command=self.generate)
        self.generate_button.grid(row=3, column=7)

        self.actions = []

        # self.c = Canvas(self.root, width=600, height=600)
        self.filename = "/home/nivg/data/balloons.png"
        self.img = Image.open(self.filename)
        self.photoimg = ImageTk.PhotoImage(self.img)
        self.c = Canvas(self.root, width=self.img.size[0], height=self.img.size[1], bg='white')
        self.myimg = self.c.create_image(0, 0, anchor=NW, image=self.photoimg)
        self.draw = ImageDraw.Draw(self.img)

        self.c.grid(row=3, columnspan=7)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.DEFAULT_COLOR
        self.choose_size_button.set(5)
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.c.bind("<ButtonPress-1>", self.on_button_press)
        self.x = self.y = 0
    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_rect(self):
        self.activate_button(self.rect_button)

    # def use_brush(self):
    #     self.activate_button(self.brush_button)

    def on_button_press(self,event):
        if self.active_button == self.rect_button:
            # save mouse drag start position
            self.start_x = self.c.canvasx(event.x)
            self.start_y = self.c.canvasy(event.y)

            # create rectangle if not yet exist
            if not self.rect:
                self.rect = self.c.create_rectangle(self.x, self.y, 1, 1, outline='red')
                self.actions.append(self.rect)


    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        for action in self.actions:
            self.c.delete(action)
        self.actions=[]
        self.img = Image.open(self.filename)
        self.draw = ImageDraw.Draw(self.img)
        self.photoimg = ImageTk.PhotoImage(self.img)
        self.c.itemconfigure(self.myimg, image=self.photoimg)
        self.rect = self.c.create_rectangle(self.x, self.y, 1, 1, outline='red')
        self.actions.append(self.rect)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = self.color
        if self.active_button == self.pen_button:
            if self.old_x and self.old_y:
                self.actions.append(self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                                   width=self.line_width, fill=paint_color,
                                   capstyle=ROUND, smooth=TRUE, splinesteps=36))
                self.draw.line([self.old_x, self.old_y, event.x, event.y], width=self.line_width, fill=paint_color)
        elif self.active_button == self.rect_button:
            curX = self.c.canvasx(event.x)
            curY = self.c.canvasy(event.y)
            self.c.coords(self.rect, self.start_x, self.start_y, curX, curY)
            # self.draw.rectangle([self.start_x, self.start_y, curX, curY], width=self.line_width, fill=paint_color)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def openfilename(self):
        # open file dialog box to select image
        # The dialogue box has a title "Open"
        filename = filedialog.askopenfilename(title='"Open')
        return filename

    def open_img(self):
        # Select the Imagename from a folder
        self.filename = self.openfilename()
        # opens the image
        for action in self.actions:
            self.c.delete(action)
        self.actions=[]
        self.img = Image.open(self.filename)
        self.draw = ImageDraw.Draw(self.img)
        self.sizes = self.img.size
        self.c.configure(width=self.img.size[0], height=self.img.size[1])
        self.photoimg = ImageTk.PhotoImage(self.img)
        self.c.itemconfigure(self.myimg, image=self.photoimg)
        self.rect = self.c.create_rectangle(self.x, self.y, 1, 1, outline='red')
        self.actions.append(self.rect)

    def save_as_png(self):
        # save postscipt image
        x = self.filename.split('.')[0] + '_edit'
        # self.c.postscript(file = x + '.eps')
        # use PIL to convert to PNG
        # img = Image.open(x + '.eps')
        # img.save(x + '.png', 'png')
        # cmd = 'convert -density 300 ' + x + '.eps ' + x + '.png'
        # os.system(cmd)
        self.img.save(x + '.png', 'png')

    def generate(self):
        self.scale = self.choose_scale_button.get()
        self.noise = float(self.noise_var.get())
        new_im = np2pt(_img_to_float32(self.img, bounds=(0.0, 1.0))).to(device=device)[:,:3,:,:]
        gt = imread(self.filename, pt=True)

        pnn = TorchPatchNN(patch_size=(7,7), dist_fn=l2_dist, batch_size=2 ** 28, bidirectional=False, alpha=1,
                           reduce='weighted_mean')
        pnn = pnn.to(device=device)
        im = gt.to(device=device)
        dtype = torch.float32
        orig_ratio = 3 / 4
        noise_std = self.noise
        noise_decay = 0.0
        depth = 15
        coords = [0, 0, 1, 1]
        if self.rect:
            coords = self.c.coords(self.rect)
            coords = [int(c) for c in coords]
        if coords != [0, 0, 1, 1]:
            mask = torch.zeros_like(im)
            mask[:,:,coords[1]:coords[3], coords[0]:coords[2]] = 1
            # new_im = im.detach().clone()
        else:
            mask = torch.ones_like(im)
        pyr = get_pyramid(im, depth=depth, ratio=orig_ratio, verbose=False)
        pyr = [lvl.to(dtype=dtype) for lvl in pyr]
        pyr = pyr[0:]
        mask_pyr = get_pyramid(mask, depth=depth, ratio=orig_ratio, verbose=False)
        mask_pyr = [lvl.to(dtype=dtype) for lvl in mask_pyr]
        mask_pyr = mask_pyr[0:]
        pnn._temperature = None
        SOFTMIN_DTPYE = torch.float64
        dst_pyr = get_pyramid(new_im, depth=len(pyr) + 1, ratio=orig_ratio, verbose=False)
        dst_pyr = [lvl.to(dtype=dtype) for lvl in dst_pyr]
        out = new_image_generation(pnn, pyr, dst_pyr, mask_pyr, top_level=self.scale, ratio=1 / orig_ratio, noise_std=noise_std, noise_decay=noise_decay, verbose=False)
        save_loc = self.filename.split('.')[0] + '_generated' + '.png'
        imwrite(save_loc, out.squeeze(0).detach().cpu())
        for action in self.actions:
            self.c.delete(action)
        self.actions=[]
        self.photoimg = ImageTk.PhotoImage(Image.open(save_loc))
        self.c.itemconfigure(self.myimg, image=self.photoimg)
        self.rect = self.c.create_rectangle(self.x, self.y, 1, 1, outline='red')
        self.actions.append(self.rect)

if __name__ == '__main__':
    Paint()