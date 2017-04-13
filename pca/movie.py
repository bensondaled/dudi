
def play_movie(m):
    # movie is called m
    imdata = pl.imshow(m[0])
    fig = pl.gcf()

    for frame in m:
        imdata.set_data(frame)
        fig.canvas.draw()
        pl.pause(0.001)
