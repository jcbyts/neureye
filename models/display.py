import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib import animation
try:
    from IPython.display import HTML
except ImportError:
    warnings.warn("Unable to import IPython.display.HTML, animshow must be called with "
                  "as_html5=False")
                  
def animshow(video, framerate=2., as_html5=True, repeat=False,
             vrange='indep1', zoom=1, title='', col_wrap=None, ax=None,
             cmap=None, plot_complex='rectangular', **kwargs):
    """Display one or more videos (3d array) as a matplotlib animation or an HTML video.

    Arguments
    ---------
    video : `np.array` or `list`
        the videos(s) to show. Videos can be either grayscale, in which case
        they must be 3d arrays of shape `(f,h,w)`, or RGB(A), in which case
        they must be 4d arrays of shape `(f,h,w,c)` where `c` is 3 (for RGB) or
        4 (to also plot the alpha channel) and `f` indexes frames. If multiple
        videos, must be a list of such arrays (note this means we do not
        support an array of shape `(n,f,h,w)` for multiple grayscale videos).
        all videos will be automatically rescaled so they're displayed at the
        same size. thus, their sizes must be scalar multiples of each other. If
        multiple videos, all must have the same number of frames (first
        dimension).
    framerate : `float`
        Temporal resolution of the video, in Hz (frames per second).
    as_html : `bool`
        If True, return an HTML5 video; otherwise return the underying matplotlib animation object
        (e.g. to save to .gif). should set to True to display in a Jupyter notebook.
    repeat : `bool`
        whether to loop the animation or just play it once
    vrange : `tuple` or `str`
        If a 2-tuple, specifies the image values vmin/vmax that are mapped to the minimum and
        maximum value of the colormap, respectively. If a string:

        * `'auto/auto1'`: all images have same vmin/vmax, which are the minimum/maximum values
                          across all images
        * `'auto2'`: all images have same vmin/vmax, which are the mean (across all images) minus/
                     plus 2 std dev (across all images)
        * `'auto3'`: all images have same vmin/vmax, chosen so as to map the 10th/90th percentile
                     values to the 10th/90th percentile of the display intensity range. For
                     example: vmin is the 10th percentile image value minus 1/8 times the
                     difference between the 90th and 10th percentile
        * `'indep1'`: each image has an independent vmin/vmax, which are their minimum/maximum
                      values
        * `'indep2'`: each image has an independent vmin/vmax, which is their mean minus/plus 2
                      std dev
        * `'indep3'`: each image has an independent vmin/vmax, chosen so that the 10th/90th
                      percentile values map to the 10th/90th percentile intensities.
    zoom : `float`
        amount we zoom the video frames (must result in an integer when multiplied by
        video.shape[1:])
    title : `str` , `list` or None
        Title for the plot:

        * if `str`, will put the same title on every plot.
        * if `list`, all values must be `str`, must be the same length as img, assigning each
          title to corresponding image.
        * if None, no title will be printed.
    col_wrap : `int` or None, optional
        number of axes to have in each row. If None, will fit all axes in a
        single row.
    ax : `matplotlib.pyplot.axis` or None, optional
        if None, we make the appropriate figure. otherwise, we resize the axes
        so that it's the appropriate number of pixels (done by shrinking the
        bbox - if the bbox is already too small, this will throw an Exception!,
        so first define a large enough figure using either make_figure or
        plt.figure)
    cmap : matplotlib colormap, optional
        colormap to use when showing these images
    plot_complex : {'rectangular', 'polar', 'logpolar'}, optional
        specifies handling of complex values.

        * `'rectangular'`: plot real and imaginary components as separate images
        * `'polar'`: plot amplitude and phase as separate images
        * `'logpolar'`: plot log_2 amplitude and phase as separate images
    kwargs :
        Passed to `ax.imshow`

    Returns
    -------
    anim : HTML object or FuncAnimation object
        Animation, format depends on `as_html`.

    """

    video = _convert_signal_to_list(video)
    video_n_frames = np.array([v.shape[0] for v in video])
    if (video_n_frames != video_n_frames[0]).any():
        raise Exception("All videos must have the same number of frames! But you "
                        "passed videos with {} frames".format(video_n_frames))
    title, vert_pct = _convert_title_to_list(title, video)
    video, title, contains_rgb = _process_signal(video, title, plot_complex, video=True)
    zooms, max_shape = _check_zooms(video, zoom, contains_rgb, video=True)
    fig, axes = _setup_figure(ax, col_wrap, video, zoom, max_shape, vert_pct)
    vrange_list, cmap = colormap_range(image=video, vrange=vrange, cmap=cmap)

    first_image = [v[0] for v in video]
    for im, a, r, t, z in zip(first_image, axes, vrange_list, title, zooms):
        _showIm(im, a, r, z, t, cmap, **kwargs)

    artists = [fig.axes[i].images[0] for i in range(len(fig.axes))]

    for i, a in enumerate(artists):
        a.set_clim(vrange_list[i])

    def animate_video(t):
        for i, a in enumerate(artists):
            frame = video[i][t].astype(np.float)
            a.set_data(frame)
        return artists

    # Produce the animation
    anim = animation.FuncAnimation(fig, frames=range(len(video[0])),
                                   interval=1000/framerate, blit=True,
                                   func=animate_video, repeat=repeat,
                                   repeat_delay=500)

    plt.close(fig)

    if as_html5:
        # to_html5_video will call savefig with a dpi kwarg, so our custom figure class will raise
        # a warning. we don't want to worry people, so we go ahead and suppress it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return HTML(anim.to_html5_video())
    return anim