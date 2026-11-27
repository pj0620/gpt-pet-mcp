"""Errors the tool layer turns into MCP tool errors. Kept free of ROS imports so main.py can be
imported (and its tool schemas listed) on a machine without rclpy."""


class RobotError(Exception):
    """A tool could not do what was asked; the message is the text the brain reads."""
