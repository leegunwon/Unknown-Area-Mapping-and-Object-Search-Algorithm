import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import numpy as np
import sys
class MapExplorer(Node):
    def __init__(self):
        super().__init__('map_explorer')
        # Parameters
        self.declare_parameter('min_goal_distance', 1.0) # 최소 이동 거리 제한
        self.min_goal_distance = self.get_parameter('min_goal_distance').value
        # Create an ActionClient for the NavigateToPose action
        self._client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        # Subscribers
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)
        self.velocity_subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.velocity_callback,
            10)
        # Initialize variables
        self.latest_map = None
        self.is_goal_reached = True
        self.is_moving = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.get_logger().info('Map Explorer node initialized')
    def map_callback(self, msg):
        """Store the latest map message"""
        self.latest_map = msg
        if not self.is_moving and self.is_goal_reached:
            self.plan_nearest_boundary_goal()
    def velocity_callback(self, msg):
        if msg.linear.x == 0.0 and msg.angular.z == 0.0:
            self.is_moving = False
            self.current_x = self.goal_x
            self.current_y = self.goal_y
            # self.get_logger().info(f'Movement completed')
    def plan_nearest_boundary_goal(self):
        """Find the nearest boundary cell that is free of obstacles and set it as the goal"""
        if self.latest_map is None:
            self.get_logger().info('No map data received yet')
            return
        width = self.latest_map.info.width
        height = self.latest_map.info.height
        resolution = self.latest_map.info.resolution
        origin_x = self.latest_map.info.origin.position.x
        origin_y = self.latest_map.info.origin.position.y
        grid = np.array(self.latest_map.data, dtype=np.int8).reshape(height, width)
        boundary_cells = []
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if grid[i, j] == -1:
                    neighbors = [grid[i-1, j], grid[i+1, j], grid[i, j-1], grid[i, j+1]]
                    if any(n == 0 for n in neighbors):
                        boundary_cells.append((i, j))
        if not boundary_cells:
            self.get_logger().info('No boundary spaces available in the map. Shutting down node.')
            self.destroy_node()
            rclpy.shutdown()
            sys.exit(1)
            return
        # Find the nearest valid boundary cell that is not near an obstacle
        min_distance = float('inf')
        nearest_cell = None
        def is_safe(i, j, safety_radius=10):
            """Check if the surrounding area of (i, j) is obstacle-free"""
            for di in range(-safety_radius, safety_radius + 1):
                for dj in range(-safety_radius, safety_radius + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        if grid[ni, nj] > 50:  # Occupied space (threshold)
                            return False
            return True
        for i, j in boundary_cells:
            cell_x = origin_x + (j * resolution)
            cell_y = origin_y + (i * resolution)
            distance = np.sqrt((cell_x - self.current_x) ** 2 + (cell_y - self.current_y) ** 2)
            if self.min_goal_distance <= distance < min_distance and is_safe(i, j):
                min_distance = distance
                nearest_cell = (i, j)
        if nearest_cell is None:
            self.get_logger().info('No valid boundary cell found within distance and safety constraints')
            self.destroy_node()
            rclpy.shutdown()
            sys.exit(1)
        goal_i, goal_j = nearest_cell
        self.goal_x = origin_x + (goal_j * resolution)
        self.goal_y = origin_y + (goal_i * resolution)
        self.send_goal(self.goal_x, self.goal_y)
        self.get_logger().info(f'Setting safe boundary exploration goal: ({self.goal_x:.2f}, {self.goal_y:.2f})')
        self.is_moving = True
        self.is_goal_reached = False
    def send_goal(self, x, y):
        """Send a goal pose to the navigation action server"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.w = 1.0
        self._client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback).add_done_callback(self.goal_done_callback)
    def feedback_callback(self, feedback):
        # self.get_logger().info(f"Feedback: {feedback}")
        pass
    def goal_done_callback(self, future):
        result = future.result()
        status = result.status
        self.is_goal_reached = True
        self.get_logger().info(f"Goal result status: {status}")
def main(args=None):
    rclpy.init(args=args)
    node = MapExplorer()
    while not node._client.wait_for_server(timeout_sec=1.0):
        node.get_logger().info('Action server not available, waiting again...')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()









