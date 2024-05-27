import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns # improves plot aesthetics
def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])
def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    # for d, (y1, y2) in zip(data[1:], ranges[1:]):
    for d, (y1, y2) in zip(data, ranges):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1) * (x2 - x1) + x1)
    return sdata
def set_rgrids(self, radii, labels=None, angle=None, fmt=None,
               **kwargs):
    """
    Set the radial locations and labels of the *r* grids.
    The labels will appear at radial distances *radii* at the
    given *angle* in degrees.
    *labels*, if not None, is a ``len(radii)`` list of strings of the
    labels to use at each radius.
    If *labels* is None, the built-in formatter will be used.
    Return value is a list of tuples (*line*, *label*), where
    *line* is :class:`~matplotlib.lines.Line2D` instances and the
    *label* is :class:`~matplotlib.text.Text` instances.
    kwargs are optional text properties for the labels:
    %(Text)s
    ACCEPTS: sequence of floats
    """
    # Make sure we take into account unitized data
    radii = self.convert_xunits(radii)
    radii = np.asarray(radii)
    rmin = radii.min()
    # if rmin <= 0:
    #     raise ValueError('radial grids must be strictly positive')
    self.set_yticks(radii)
    if labels is not None:
        self.set_yticklabels(labels)
    elif fmt is not None:
        self.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    if angle is None:
        angle = self.get_rlabel_position()
    self.set_rlabel_position(angle)
    for t in self.yaxis.get_ticklabels():
        t.update(kwargs)
    return self.yaxis.get_gridlines(), self.yaxis.get_ticklabels()
class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=6):
        angles = np.arange(0, 360, 360./len(variables))
        axes = [fig.add_axes([0.01,0,0.85,0.9],polar=True,
                label = "axes{}".format(i))
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles,
                                         labels=variables)
        [txt.set_rotation(angle-90) for txt, angle
             in zip(text, angles)]

        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)

        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i],
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2))
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            gridlabel[1] = ""
            # ax.set_rgrids(grid, labels=gridlabel, angle=angles[i])

            set_rgrids(ax, grid, labels=gridlabel, angle=angles[i], fontsize=15)
            #ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
            ax.spines['polar'].set_visible(False)
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

# variables = ["" for i in range(17)]

# performance of random selection
line_a = (0.37764,0.0568,0.1074,0.1693,0.11468,0.55578,35.49358,3.76942,2.10502,2.30388,0.16104,0.1753,4.84224,0.02278,0.053516,23.17746,0.24712)

# performance of all selection
line_b = (0.3624,0.05866,0.10546,0.16782,0.10988,0.54912,34.17616,2.72852,2.0902,2.24476,0.15904,0.17416,4.7682,0.0223,0.05286,23.56868,0.24916)

# performance of single module
line_c = (0.37416,0.05748,0.107172,0.17062,0.11332,0.54244,34.3332,1.9829,2.0595,2.2734,0.16538,0.17616,4.77844,0.022322,0.056166,23.12392,0.24694)

# main performance
line2 = (0.35146,0.05544,0.10516,0.16786,0.11294,0.56008,34.27586,1.9999,2.12144,2.27702,0.16088,0.17384,4.77016,0.022046,0.0521,23.39414,0.24516)


# draw radar-random
variables = ["" for i in range(17)]
ranges = []
# for em, m, ec, c in zip(EvMaple_data, Maple_data, EvClip_data, Clip_data):
for a,b in zip(line_a, line2):
    range_min = min(a,b)
    range_max = max(a,b)
    if a<b:
        ranges.append((range_min*0.8, range_max))
    else:
        ranges.append((range_min*0.9, range_max))

fig1 = plt.figure(figsize=(10, 10), dpi=500)

radar = ComplexRadar(fig1, variables, ranges)

radar.plot(line_a,linewidth=2)
radar.fill(line_a, alpha=0.15)
radar.plot(line2, linewidth=2)
radar.fill(line2, alpha=0.3)
# plt.show()
fig1.savefig('radar_random.png',dpi=500)
plt.close(fig1)

# draw radar-all
variables = ["" for i in range(17)]
ranges = []
# for em, m, ec, c in zip(EvMaple_data, Maple_data, EvClip_data, Clip_data):
for a,b in zip(line_b, line2):
    range_min = min(a,b)
    range_max = max(a,b)
    if a<b:
        ranges.append((range_min*0.8, range_max))
    else:
        ranges.append((range_min*0.9, range_max))

fig2 = plt.figure(figsize=(10, 10), dpi=500)

radar = ComplexRadar(fig2, variables, ranges)

radar.plot(line_b,linewidth=2,color='purple')
radar.fill(line_b, alpha=0.15,color='purple')
radar.plot(line2, linewidth=2,color='orange')
radar.fill(line2, alpha=0.3,color='orange')
# plt.show()
fig2.savefig('radar_all.png',dpi=500)
plt.close(fig2)

# draw radar-single
variables = ["" for i in range(17)]
ranges = []
# for em, m, ec, c in zip(EvMaple_data, Maple_data, EvClip_data, Clip_data):
for a,b in zip(line_c, line2):
    range_min = min(a,b)
    range_max = max(a,b)
    if a<b:
        ranges.append((range_min*0.8, range_max))
    else:
        ranges.append((range_min*0.9, range_max))

fig3 = plt.figure(figsize=(10, 10), dpi=500)


radar = ComplexRadar(fig3, variables, ranges)

radar.plot(line_c,linewidth=2,color='darkgreen')
radar.fill(line_c, alpha=0.15,color='darkgreen')
radar.plot(line2, linewidth=2,color='orange')
radar.fill(line2, alpha=0.3,color='orange')
# plt.show()
fig3.savefig('radar_single.png',dpi=500)
plt.close(fig3)
