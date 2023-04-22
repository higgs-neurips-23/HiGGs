#!/bin/bash
https://drive.google.com/file/d/1VWoU_gWLjePHpvt-ZY-a0ZoDG7YYpY-3/view?usp=sharing
echo "Downloading .zip"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VWoU_gWLjePHpvt-ZY-a0ZoDG7YYpY-3" -O saved_models.zip && rm -rf /tmp/cookies.txt
echo "Unpacking zip"
unzip saved_models.zip
echo "Removing .zip"
rm saved_models.zip


