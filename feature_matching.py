import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        
        # QoS 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.subscription = self.create_subscription(
            CompressedImage,
            '/oakd/rgb/preview/image_raw/compressed',
            self.image_callback,
            qos_profile)
        
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/pose',
            self.pose_callback,
            10  # 큐 사이즈
        )
        self.pose_subscription  # prevent unused variable warning

        self.bridge = CvBridge()
        self.last_callback_time = time.time()

        self.reference_img1 = cv2.imread("/home/su/ws_driving2/src/explorer/explorer/man_orig.png")
        self.reference_img2 = cv2.imread("/home/su/ws_driving2/src/explorer/explorer/ext_orig.png")
        self.reference_gray1 = cv2.cvtColor(self.reference_img1, cv2.COLOR_BGR2GRAY)
        self.reference_gray2 = cv2.cvtColor(self.reference_img2, cv2.COLOR_BGR2GRAY)
        
        # SIFT 생성
        self.sift = cv2.SIFT_create(nfeatures=5000)
        self.kp1, self.des1 = self.sift.detectAndCompute(self.reference_gray1, None)
        self.kp2, self.des2 = self.sift.detectAndCompute(self.reference_gray2, None)
        
        # BFMatcher를 L2로 설정
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)



        self.K = np.array([[344.4533, 0, 124.346],
                           [0, 344.4533, 127.2864],
                           [0, 0, 1]], dtype=np.float32)

        # self.K = np.array([[202.6196, 0, 124.346],
        #                    [0, 202.6196, 127.2864],
        #                    [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([-3.475167, -38.573474, 0.000343, -9.3772e-05, 286.440093, -3.640805, -36.688980, 279.052368], dtype=np.float32)
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        self.posemap = None
        
        # 카메라 좌표계에서 베이스 좌표계로의 변환 행렬
        self.T_base_cam = np.array([[0, 0, 1, -0.060],
                                       [-1, 0, 0, 0],
                                       [0, -1, 0, 0.244],
                                       [0, 0, 0, 1]], dtype=np.float32)

        # TF 브로드캐스터 초기화
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.get_logger().info("ImageSubscriber node has been started")


    def quaternion_matrix(self,quaternion):
        """
        Converts a quaternion (x, y, z, w) into a 4x4 rotation matrix.
        Also handles small quaternions by returning an identity matrix.

        :param quaternion: A tuple or list of length 4 (x, y, z, w).
        :return: A 4x4 numpy array representing the rotation matrix.
        """
        x, y, z, w = quaternion
        q = np.array([w,x, y, z], dtype=np.float64, copy=True)

        # Compute the norm (magnitude) of the quaternion
        n = np.dot(q, q)

        # If the norm is very small, return the identity matrix
        if n < np.finfo(q.dtype).eps:
            return np.identity(4)

        # Normalize the quaternion
        q *= np.sqrt(2.0 / n)

        # Calculate the outer product of the quaternion with itself
        q_outer = np.outer(q, q)

        # Construct the 4x4 rotation matrix
        mat = np.array([
            [1.0 - q_outer[2, 2] - q_outer[3, 3], q_outer[1, 2] - q_outer[3, 0], q_outer[1, 3] + q_outer[2, 0], 0.0],
            [q_outer[1, 2] + q_outer[3, 0], 1.0 - q_outer[1, 1] - q_outer[3, 3], q_outer[2, 3] - q_outer[1, 0], 0.0],
            [q_outer[1, 3] - q_outer[2, 0], q_outer[2, 3] + q_outer[1, 0], 1.0 - q_outer[1, 1] - q_outer[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        return mat
 


    def create_transformation_matrix(self,rvec, tvec):
        """
        PnP로 얻은 rvec, tvec을 이용하여 4x4 변환 행렬 T를 생성하는 함수.

        :param rvec: (3x1) 회전 벡터 (Rodrigues 회전 벡터)
        :param tvec: (3x1) 이동 벡터
        :return: (4x4) 변환 행렬 T
        """
        # 회전 벡터를 회전 행렬로 변환
        R, _ = cv2.Rodrigues(rvec)

        # 4x4 변환 행렬 생성
        T = np.eye(4)
        T[:3, :3] = R  # 회전 행렬
        # T[:3, 3] = tvec.flatten()*1.7  # 이동 벡터
        T[:3, 3] = tvec.flatten()  # 이동 벡터

        return T

    def cam_to_base_transform(self, tvec):
        # tvec이 리스트인 경우 numpy 배열로 변환
        if isinstance(tvec, list):
            tvec = np.array(tvec)
        
        # tvec이 1차원 배열인지 확인하고 변환
        if tvec.ndim > 1:
            tvec = tvec.flatten()
            
        # 카메라 좌표계의 위치를 베이스 좌표계로 변환
        t_cam = np.array([tvec[0], tvec[1], tvec[2], 1.0])
        t_base = self.T_cam_to_base @ t_cam
        return t_base[:3]
        
    def pose_callback(self, msg):
        """
        /pose 토픽에서 수신된 PoseWithCovarianceStamped 메시지를 처리하여 posemap에 저장.
        """
        self.posemap = msg.pose.pose  # 변환 행렬을 만들기 위한 Pose 저장
        self.get_logger().info(f"Pose received and stored: {self.posemap}")
        

    def create_transformation_matrix_from_pose(self):
        """
        Pose 메시지에서 위치와 회전 정보를 사용하여 4x4 변환 행렬 생성.

        :return: 4x4 변환 행렬
        """
        if self.posemap is None:
            self.get_logger().warn("posemap is not yet set.")
            return None
        
        pose = self.posemap
        
        # 위치 정보
        position = pose.position
        tvec = np.array([position.x, position.y, position.z])

        # 쿼터니언 회전 정보
        orientation = pose.orientation
        q = [orientation.x, orientation.y, orientation.z,orientation.w ]

        # 자체 quaternion_matrix 함수를 이용해 회전 행렬로 변환
        R = self.quaternion_matrix(q)[:3, :3]

        # 4x4 변환 행렬
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec

        self.get_logger().info(f'Transformation Matrix:\n{T}')
        return T
    def publish_marker(self, rvec, tvec, color):
        if self.posemap is None:
            self.get_logger().warn("posemap is not yet set.")
            return None

        T_cam_img = self.create_transformation_matrix(rvec, tvec) 
        # rvec, tvec = self.cam_pnp_transform(rvec, tvec)
        T_base_img = self.T_base_cam @ T_cam_img
        T_map_base = self.create_transformation_matrix_from_pose()
        T_map_img = T_map_base @ T_base_img

        # Extract translation and rotation from T_map_img
        rotation_matrix = T_map_img[:3, :3]
        translation = T_map_img[:3, 3]
        quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)

        # Create a marker
        scale_x = 0.3
        scale_y = 0.3
        scale_z = 0.05
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "transform"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = translation[0] + scale_x/2
        marker.pose.position.y = translation[1] + scale_y/2
        marker.pose.position.z = translation[2] + scale_z/2
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]
        marker.scale.x = scale_x  
        marker.scale.y = scale_y
        marker.scale.z = scale_z
        marker.color.a = 1.0
        if color == 'red':
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

        elif color == 'blue':
            marker.color.b = 1.0
            marker.color.g = 0.0
            marker.color.r = 0.0

        # Publish the marker
        self.marker_pub.publish(marker)
        self.get_logger().info("Marker published")

    def rotation_matrix_to_quaternion(self, R):
        """
        Convert a rotation matrix to a quaternion.

        :param R: 3x3 rotation matrix
        :return: quaternion [x, y, z, w]
        """
        q = np.empty((4, ))
        t = np.trace(R)
        if t > 0.0:
            t = np.sqrt(t + 1.0)
            q[3] = 0.5 * t
            t = 0.5 / t
            q[0] = (R[2, 1] - R[1, 2]) * t
            q[1] = (R[0, 2] - R[2, 0]) * t
            q[2] = (R[1, 0] - R[0, 1]) * t
        else:
            i = 0
            if (R[1, 1] > R[0, 0]):
                i = 1
            if (R[2, 2] > R[i, i]):
                i = 2
            j = (i + 1) % 3
            k = (i + 2) % 3
            t = np.sqrt((R[i, i] - R[j, j] - R[k, k]) + 1.0)
            q[i] = 0.5 * t
            t = 0.5 / t
            q[3] = (R[k, j] - R[j, k]) * t
            q[j] = (R[j, i] + R[i, j]) * t
            q[k] = (R[k, i] + R[i, k]) * t

        # Normalize the quaternion to avoid numerical errors
        norm = np.linalg.norm(q)
        q = q / norm

        return q





    def image_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_callback_time < 0.5:
            return
        self.last_callback_time = current_time

        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp3, des3 = self.sift.detectAndCompute(gray, None)
        

        def process_object(kp_ref, des_ref, reference_width_m,row_pixcel , color):
            # Ensure the descriptors are of the same type and size
            if des_ref is None or des3 is None:
                return
            matches = self.bf.knnMatch(des_ref, des3, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.4 * n.distance]
            
            if len(good_matches) >= 10:
                object_points = []
                image_points = []

                # 물체의 실제 크기와 이미지 크기를 비교하여 3D 좌표로 변환
                for m in good_matches:
                    # 참고 이미지의 각 특징점의 위치
                    pt_ref = np.array(kp_ref[m.queryIdx].pt, dtype=np.float32)
                    
                    # 물체의 실제 크기와 이미지의 크기 비율을 계산
                    scale_factor = reference_width_m / row_pixcel
                    object_points.append(np.array([pt_ref[0] * scale_factor, pt_ref[1] * scale_factor, 0], dtype=np.float32))
                    
                    # 이미지 상의 특징점 위치
                    pt_img = np.array(kp3[m.trainIdx].pt, dtype=np.float32)
                    image_points.append(pt_img)
                if color == 'red':
                    img_matches = cv2.drawMatches(self.reference_img1, kp_ref, frame, kp3, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                elif color == 'blue':
                    img_matches = cv2.drawMatches(self.reference_img2, kp_ref, frame, kp3, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                cv2.imshow(f'Matches ({color})', img_matches)
                cv2.waitKey(1)  # 이미지 업데이트를 위해 잠시 대기

                object_points = np.float32(object_points)
                image_points = np.float32(image_points)
                
                success, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, self.K, self.dist_coeffs)
                if success:
                    self.publish_marker(rvec, tvec, color)

                # 특징점 시각화 (매칭된 특징점들 연결)

        process_object(self.kp1, self.des1, 0.18, 680,'red')
        process_object(self.kp2, self.des2, 0.18, 680,'blue')


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
