import os

def area(x1, y1, x2, y2, x3, y3):
    return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0)

def is_point_in_quadrilateral_area_method(x, y, quad):
    x1, y1 = quad[0]
    x2, y2 = quad[1]
    x3, y3 = quad[2]
    x4, y4 = quad[3]

    # Diện tích của tứ giác
    area_quad = area(x1, y1, x2, y2, x3, y3) + area(x1, y1, x4, y4, x3, y3)

    # Tổng diện tích của 4 tam giác được tạo bởi điểm (x, y)
    area_p = (area(x, y, x1, y1, x2, y2) +
              area(x, y, x2, y2, x3, y3) +
              area(x, y, x3, y3, x4, y4) +
              area(x, y, x4, y4, x1, y1))

    # Nếu tổng diện tích bằng diện tích tứ giác thì điểm nằm trong tứ giác
    return area_quad == area_p

# Ví dụ sử dụng
# quad = [(0, 0), (4, 0), (4, 4), (0, 4)]
# point = (2, 2)

# print(is_point_in_quadrilateral_area_method(point[0], point[1], quad))  # Kết quả: True
