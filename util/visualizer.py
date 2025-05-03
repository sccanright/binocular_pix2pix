import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE


try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256, use_wandb=False):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these images (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    # Extract the run directory name (e.g., "run_22" or "run_41")
    run_dir = os.path.basename(os.path.dirname(image_path))
    
    # Get the base image directory from the webpage
    base_image_dir = webpage.get_image_dir()
    if not base_image_dir:
        raise ValueError("webpage.get_image_dir() returned None or an empty string.")

    # Modify image_dir to include the run directory name
    image_dir = os.path.join(base_image_dir, run_dir)
    os.makedirs(image_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    webpage.add_header(f"Images from: {run_dir}")
    ims, txts, links = [], [], []
    ims_dict = {}

        ###print(f"Function called! Image path: {image_path}")
        ###print(f"Available visuals keys: {visuals.keys()}")
    for label, im_data in visuals.items():
            ###print(f"Processing {label}, shape: {im_data.shape}")
        ## This handles my focus stack of 10 images. Saving each image with 3 channels or 1 channel depending on style (rgb or grayscale)
        #_________________________________________________________________________________#
        if im_data.shape[1] == 30:  # Check if this is a focus stack 10 rgb images (30 channels)
            # Group every 3 channels into one RGB image
            for group_idx in range(10):
                rgb_tensor = im_data[:, group_idx * 3: group_idx * 3 + 3, :, :]  # Extract 3 channels for rgb
                rgb_image = util.tensor2im(rgb_tensor)  # Convert to rgb image
                group_name = f"{label}_{group_idx+1}.png"
                save_path = os.path.join(image_dir, group_name)
                util.save_image(rgb_image, save_path, aspect_ratio=aspect_ratio)
                
                # Include the run directory in the relative path for the webpage
                relative_path = os.path.join(run_dir, group_name)
                ims.append(relative_path)
                txts.append(f"{label}_{group_idx+1}")
                links.append(relative_path)
                if use_wandb:
                    ims_dict[f"{label}_{group_idx+1}"] = wandb.Image(rgb_image)
                    
        elif im_data.shape[1] == 10:    # Check for focus stack of 10 grayscale images (10 channels)
                # Process each grayscale image
                for group_idx in range(10):
                    grayscale_tensor = im_data[:, group_idx, :, :]  # Extract 1 channel for grayscale
                    grayscale_image = util.tensor2im(grayscale_tensor) # Convert to grayscale image
                    group_name = f"{label}_{group_idx+1}.png"
                    save_path = os.path.join(image_dir, group_name)
                    util.save_image(grayscale_image, save_path, aspect_ratio=aspect_ratio)
                    
                    # Include the run directory in the relative path for the webpage
                    relative_path = os.path.join(run_dir, group_name)
                    ims.append(relative_path)
                    txts.append(f"{label}_{group_idx+1}")
                    links.append(relative_path)
                    if use_wandb:
                        ims_dict[f"{label}_{group_idx+1}"] = wandb.Image(grayscale_image)
                            
        else:  # For standard rgb or grayscale images
            im = util.tensor2im(im_data) # Convert tensor to image
            image_name = f"{label}.png"
            save_path = os.path.join(image_dir, image_name)
            util.save_image(im, save_path, aspect_ratio=aspect_ratio)
            
            # Include the run directory in the relative path for the webpage
            relative_path = os.path.join(run_dir, image_name)
            ims.append(relative_path)
            txts.append(label)
            links.append(relative_path)
            if use_wandb:
                ims_dict[label] = wandb.Image(im)
        #_________________________________________________________________________________#
        
        ## This was what was here originally..
        ### Saves a single grayscale image for each channel (30 grayscale images)
        #_________________________________________________________________________________#
        #import torch
            ###print(torch.max(im_data))
            ###print(torch.min(im_data))
        #im = util.tensor2im(im_data)
            ###print(np.max(im))
            ###print(np.min(im))
        #image_name = '%s_%s.png' % (name, label)
        #save_path = os.path.join(image_dir, image_name)
        #util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        #ims.append(image_name)
        #txts.append(label)
        #links.append(image_name)
        #if use_wandb:
            #ims_dict[label] = wandb.Image(im)
        #_________________________________________________________________________________#

    webpage.add_images(ims, txts, links, width=width)
    if use_wandb:
        wandb.log(ims_dict)

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.wandb_project_name = opt.wandb_project_name
        self.current_epoch = 0
        self.ncols = opt.display_ncols

        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_wandb:
            self.wandb_run = wandb.init(project=self.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
            self.wandb_run._label(repo='CycleGAN-and-pix2pix')

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    #images.append(image_numpy.transpose([2, 0, 1]))
                    images.append(image_numpy)
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                #white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                #print(f'image numpy: {image_numpy.shape}')
                white_image = np.ones_like(image_numpy) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    #print(f'images shape before vis images: {[image.shape for image in images]}')  # Print the shape of each image in the list
                    # make rbg for visualizer
                    vis_images = []
                    for image in images:
                        if len(image.shape) == 4:
                            image = image[:, 0, :, :]
                            #print(f'size reduced at visualizer ln 160 {image.shape}')
                        if image.shape[0] > 3:
                            image = np.mean(image, axis=0)
                            image = np.tile(image, (3, 1, 1))  # Repeat the image across the RGB channels
                        vis_images.append(image)

                    #print(f'images shape before vis images reduced size: {[vis_image.shape for vis_image in vis_images]}')  # Print the shape of each image in the list
                    self.vis.images(vis_images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_wandb:
            columns = [key for key, _ in visuals.items()]
            columns.insert(0, 'epoch')
            result_table = wandb.Table(columns=columns)
            table_row = [epoch]
            ims_dict = {}
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                wandb_image = wandb.Image(image_numpy)
                table_row.append(wandb_image)
                ims_dict[label] = wandb_image
            self.wandb_run.log(ims_dict)
            if epoch != self.current_epoch:
                self.current_epoch = epoch
                result_table.add_data(*table_row)
                self.wandb_run.log({"Result": result_table})

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()
        if self.use_wandb:
            self.wandb_run.log(losses)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
