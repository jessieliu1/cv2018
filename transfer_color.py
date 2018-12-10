from color_transfer import color_transfer
import cv2


def transfer_color(im_source, im_target):
    source = cv2.imread(im_source)
    target = cv2.imread(im_target)
    transfer = color_transfer(source, target)
    cv2.imwrite("output.png", transfer)
    cv2.imshow("output.png", transfer)



def main():
    transfer_color("van_gogh.jpg", "pikachu.jpg")

main()