# mandelbrot_zoom.mp4:
# 	ffmpeg -r 30 -i images/mandelbrot_zoom_frame_%d.png -c:v libx264 -pix_fmt yuv420p $@

# mandelbrot_zoom.gif:
# 	ffmpeg -r 30 -i images/mandelbrot_zoom_frame_%d.png -vcodec gif -y $@

# mandelbrot_iter_interpolate.mp4:
# 	ffmpeg -r 5 -i images/mandelbrot_iter_frame_%d.png -filter:v "minterpolate='fps=30'" -c:v libx264 -pix_fmt yuv420p $@

# mandelbrot_iter.mp4:
# 	ffmpeg -r 5 -i images/mandelbrot_iter_frame_%d.png -c:v libx264 -pix_fmt yuv420p $@

# mandelbrot_iter.gif:
# 	ffmpeg -r 5 -i images/mandelbrot_iter_frame_%d.png -vcodec gif -y $@

mandelbrot-iter.mp4: mandelbrot-iter.gif
	ffmpeg -r 5 -i mandelbrot-iter.gif -vcodec libx264 -pix_fmt yuv420p mandelbrot-iter.mp4
