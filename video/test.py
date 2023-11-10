from utils_track import correct_ids, write_track, del_not_human, show_boxes, change_numbers

track = correct_ids('track.txt',
                    5,
                    25,
                    1000,
                    10000,
                    8,
                    0.9985,
                    1,
                    5,
                    40,
                    )

# delete not human detections
track = del_not_human(track, 0.985, 5)

# show missed human detections
track = show_boxes(track)

track = change_numbers(track)

write_track(track)

