from matplotlib.colors import LinearSegmentedColormap


bluegreen = LinearSegmentedColormap('bluegreen', {
    'red': ((0., 0., 0.),
            (1., 0., 0.)),
    'green': ((0., 0., 0.),
              (1., 1., 1.)),
    'blue': ((0., 0.2, 0.2),
             (0.5, 0.5, 0.5),
             (1., 0., 0.))
    })
