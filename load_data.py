input_csv = "/mnt/data2t/datasets/UTCDA/videos_splited/train_clip.csv"
output_csv = "/mnt/data/quangtungbk/Continuous-Action-Recognition-v2/train_clip_rear.csv"

with open(input_csv, "r") as fin, open(output_csv, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        path, label = line.split(",")
        if "_rear.MP4" in path:
            fout.write(line + "\n")

print("Done! Front-view data saved to:", output_csv)