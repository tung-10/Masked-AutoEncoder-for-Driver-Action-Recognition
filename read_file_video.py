from decord import VideoReader, cpu

data_path = "rgb1_fixed.avi"
vr = VideoReader(data_path, num_threads=1, ctx=cpu(0))
print(vr)