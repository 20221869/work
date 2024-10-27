import cv2
import numpy as np

class ImageMasker:
    def __init__(self, image_path):
        # 读取图片
        self.image = cv2.imread(image_path)
        height, width = self.image.shape[:2]
        max_width = 800
        max_height = 600
        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            self.image = cv2.resize(self.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.points = []
        self.drawing = False
        self.colors_to_detect = []  # 存储用户点击的颜色

    def draw_curve(self, event, x, y, flags, param):
        """鼠标事件回调函数，用于绘制曲线区域"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.points.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.points:
                # 完成绘制，填充当前点到掩膜
                cv2.fillPoly(self.mask, [np.array(self.points)], (255, 255, 255))

    def select_region(self):
        """让用户手动选择一个不规则封闭区域"""
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.draw_curve)

        print("请在图片上按左键绘制曲线，松开左键结束绘制，按ESC键退出。")
        while True:
            image_copy = self.image.copy()
            if len(self.points) > 0:
                cv2.polylines(image_copy, [np.array(self.points)], False, (0, 255, 0), 2)
            cv2.imshow('Image', image_copy)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # 按ESC键退出
                break

        # 在窗口关闭前填充最后的轮廓
        if len(self.points) > 2:
            cv2.fillPoly(self.mask, [np.array(self.points)], (255, 255, 255))

        cv2.destroyAllWindows()

    def crop_to_selected_region(self):
        """根据手动选择的区域裁剪图片"""
        if not self.points:
            raise ValueError("尚未选择区域，请先调用 select_region 方法。")

        # 获取掩膜的非零区域，即选定的区域
        masked_image = cv2.bitwise_and(self.image, self.image, mask=self.mask)
        return masked_image

    def detect_color_blocks(self, image):
        """识别与存储颜色匹配的色块并显示其轮廓"""
        for color in self.colors_to_detect:
            lower_bound = np.array([color[0] - 10, color[1] - 10, color[2] - 10])
            upper_bound = np.array([color[0] + 10, color[1] + 10, color[2] + 10])
            mask = cv2.inRange(image, lower_bound, upper_bound)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 0, 255), 1)  # 红色轮廓
        return image

    def show_result(self):
        """显示处理结果的图片并允许用户点击存储颜色"""
        result = self.crop_to_selected_region()

        cv2.namedWindow('Processed Image')
        cv2.setMouseCallback('Processed Image', self.store_color)

        print("请在图片上点击不同的颜色点存储颜色，按ESC键退出。")
        while True:
            result_copy = result.copy()
            cv2.imshow('Processed Image', result_copy)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # 按ESC键退出
                break

        cv2.destroyAllWindows()

    def store_color(self, event, x, y, flags, param):
        """存储用户点击的颜色"""
        if event == cv2.EVENT_LBUTTONDOWN:
            result = self.crop_to_selected_region()
            color = result[y, x]
            self.colors_to_detect.append(color)
            print(f"存储颜色: {color}")

    def calculate_paint_mixture(self, rgb_color):
        """计算颜料混合比例"""
        # 基础颜色（红、绿、蓝）的RGB表示
        base_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255)
        ]

        # 计算与基础颜色的欧几里得距离
        distances = [np.linalg.norm(np.array(rgb_color) - np.array(base_color)) for base_color in base_colors]

        # 归一化距离以获得混合比例
        total_distance = sum(distances)
        if total_distance == 0:
            return [1.0, 0.0, 0.0]
        mix_ratios = [1 - (dist / total_distance) for dist in distances]

        return mix_ratios

def main():
    image_path = '/media/yogoe/UBUNTU 22_0/cake2.jpg'  # 替换为你的图片路径

    # 创建 ImageMasker 对象并选择区域
    masker = ImageMasker(image_path)
    masker.select_region()

    # 显示处理后的图片并允许用户存储颜色
    masker.show_result()

    # 显示最终的处理结果（包含色块轮廓）
    result = masker.crop_to_selected_region()
    result = masker.detect_color_blocks(result)

    # 计算颜料混合比例
    for color in masker.colors_to_detect:
        mix_ratios = masker.calculate_paint_mixture(color)
        print(f"颜色 {color} 的颜料混合比例: 红: {mix_ratios[0]:.2f}, 绿: {mix_ratios[1]:.2f}, 蓝: {mix_ratios[2]:.2f}")

    cv2.imshow("Final Processed Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()