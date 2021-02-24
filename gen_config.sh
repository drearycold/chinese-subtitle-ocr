FRAMES_PER_PART=27000
VIDEOFILE=$1
FC=`mediainfo --Inform='Video;%FrameCount%' "$VIDEOFILE"`
I=2

while :
do
    START=$((FRAMES_PER_PART*I))
    END=$((FRAMES_PER_PART*(I+1)))


    if [ $START -gt $FC ]
    then
        break
    fi
    if [ $END -gt $FC ]
    then
        END=$FC
    fi

    echo $START $END

    cat config_part.yml | sed \
        -e "s#^video:.*#video: \"$VIDEOFILE\"#" \
        -e "s/^video_offset_start:.*/video_offset_start: $START/" \
        -e "s/^video_offset_end:.*/video_offset_end: $END/" \
        -e "s#^output_sub_video:.*#output_sub_video: \"output/output_sub_${I}_video.mp4\"#" \
        -e "s#^output_sub_ocr:.*#output_sub_ocr: \"output/output_sub_${I}_ocr.json\"#" \
        > config/config_part_$I.yml

    I=$((I+1))
done
