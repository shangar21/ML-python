import cairo
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

SHAPE_LABELS = ['rectangle', 'circle', 'triangle']
COLOUR_LABELS = list('rgb')

def make_rectangle(min_size, max_size, img_size, cr):
    w, h = np.random.randint(min_size, max_size, size=2)
    x = np.random.randint(0, img_size - w)
    y = np.random.randint(0, img_size - h)
    cr.rectangle(x, y, w, h)
    return [x, y, w, h]

def make_circle(min_size, max_size, img_size, cr):
    r = 0.5 * np.random.randint(min_size, max_size)
    x = int(np.random.uniform(r, img_size - r))
    y = int(np.random.uniform(r, img_size - r))
    cr.arc(x, y, r, 0, 2*np.pi)
    return [x-r, y-r, 2*r, 2*r]

def make_triangle(min_size, max_size, img_size, cr):
    w, h = np.random.randint(min_size, max_size, size=2)
    x = np.random.randint(0, img_size - w)
    y = np.random.randint(0, img_size - h)
    cr.move_to(x, y)
    cr.line_to(x+w, y)
    cr.line_to(x+w, y+h)
    cr.line_to(x, y)
    cr.close_path()
    return [x, y, w, h]

def generate_objects(num_imgs, img_size, min_size, max_size, num_objects):
    bboxes = np.zeros((num_imgs, num_objects, 4))
    imgs = np.zeros((num_imgs, img_size, img_size, 4), dtype=np.uint8)
    shapes = np.zeros((num_imgs, num_objects), dtype=int)
    colors = np.zeros((num_imgs, num_objects), dtype=int)
    shape_makers = [make_rectangle, make_circle, make_triangle]

    for i_img in range(num_imgs):
        surface = cairo.ImageSurface.create_for_data(imgs[i_img], cairo.FORMAT_ARGB32, img_size, img_size)
        cr = cairo.Context(surface)
        cr.set_source_rgb(1, 1, 1)
        cr.paint()

        for i_object in range(num_objects):
            shape = np.random.randint(len(SHAPE_LABELS))
            shapes[i_img, i_object] = shape
            bboxes[i_img, i_object] = shape_makers[shape](min_size, max_size, img_size, cr)
            color = np.random.randint(len(COLOUR_LABELS))
            colors[i_img, i_object] = color
            max_offset = 0.3
            r_offset, g_offset, b_offset = max_offset * 2. * (np.random.rand(3) - 0.5)
            if color == 0:
                cr.set_source_rgb(1-max_offset+r_offset, 0+g_offset, 0+b_offset)
            elif color == 1:
                cr.set_source_rgb(0+r_offset, 1-max_offset+g_offset, 0+b_offset)
            elif color == 2:
                cr.set_source_rgb(0+r_offset, 0-max_offset+g_offset, 1+b_offset)
            cr.fill()
    return bboxes, imgs, shapes, colors

def plot_objects(bboxes, imgs, shapes, colors, idx, plt):
    img_size = imgs.shape[1]
    plt.imshow(imgs[idx], interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
    print(bboxes)
    for bbox, shape, color in zip(bboxes[idx], shapes[idx], colors[idx]):
        plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='k', fc='none'))
        plt.annotate(SHAPE_LABELS[shape], (bbox[0], bbox[1] + bbox[3] + 0.7), color=COLOUR_LABELS[color], clip_on=False)
    plt.show()

if __name__ == '__main__':
    bboxes, imgs, shapes, colors = generate_objects(10, 32, 4, 16, 2)
    plot_objects(bboxes, imgs, shapes, colors, 2, plt)
