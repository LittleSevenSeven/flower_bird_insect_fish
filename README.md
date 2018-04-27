# flower_bird_insect_fish
    in this crawl pictures of flower, bird, insect and fish from the internet, then save the figures to mysql database, finally get the fiugures from the database and classify the pictures.<br>
    *`crawl.py`爬取图片，`saveImg.py`将图片存入mysql<br>
    *`readImg.py`从mysql中取出图片，`to_npy(label).py`和`to_npy(image).py`将图片处理为CNN模型可用的格式，并打好标签<br>
    *`LeNet.py`, `AlexNet.py`, `VGGV16.py`和`VGG19.py`建立CNN模型，分类图片。
