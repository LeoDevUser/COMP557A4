from helperclasses import Ray
import taichi as ti
from pyglm import glm
import taichi.math as tm
import random

@ti.data_oriented
class Camera:
    def __init__(self, width, height, eye_position:glm.vec3, lookat:glm.vec3, up:glm.vec3, fovy) -> None:
        '''	Initialize the camera with given parameters.
		Args:
			width (int): width of the image in pixels
			height (int): height of the image in pixels
			eye_position (glm.vec3): position of the camera in world space
			lookat (glm.vec3): point the camera is looking at
			up (glm.vec3): up direction for the camera
			fovy (float): vertical field of view in degrees
		Returns:
			None

			Note that while glm types are provided to construct the class, the 
			member variables must be of taich tm.vec3 types so that they used 
			in the create_ray Taichi function.  
		'''
        self.width = width
        self.height = height		
        self.distance_to_plane = 1.0

        # TODO: Objective 1: Compute camera frame basis vectors, and top bottom left right for ray generation
        # NOTE: glm vectors are passed in to permit the work to be done here in python, but stored vectors must be tm.vec3

        # Compute proper values below!
        w_glm = glm.normalize(eye_position - lookat)
        self.w = tm.vec3(w_glm.x, w_glm.y, w_glm.z)
        u_glm = glm.normalize(glm.cross(up, w_glm))
        self.u = tm.vec3(u_glm.x, u_glm.y, u_glm.z)
        v_glm = glm.cross(w_glm, u_glm)
        self.v = tm.vec3(v_glm.x, v_glm.y, v_glm.z)

        #convert fovy from degrees to radians and compute viewport dimensions
        fovy_rad = glm.radians(fovy)
        height_at_plane = 2.0 * self.distance_to_plane * glm.tan(fovy_rad / 2.0)
        width_at_plane = height_at_plane * (width / height)

        #set viewport bounds
        self.top = height_at_plane / 2.0
        self.bottom = -height_at_plane / 2.0
        self.right = width_at_plane / 2.0
        self.left = -width_at_plane / 2.0

        # Store eye position in world space
        self.eye_position = tm.vec3(eye_position.x, eye_position.y, eye_position.z)

    @ti.func
    def create_ray(self, x, y, jitter=False) -> Ray:
        ''' Create a ray going through pixel (x,y) in image space '''

        #normalize pixel coordinates to [0, 1]
        u_norm = x / self.width
        v_norm = y / self.height

        if jitter:
            u_norm += (ti.random() / self.width)
            v_norm += (ti.random() / self.height)

        #map to viewport coordinates
        u_coord = self.left + u_norm * (self.right - self.left)
        v_coord = self.bottom + v_norm * (self.top - self.bottom)

        direction = u_coord * self.u +v_coord * self.v -self.distance_to_plane * self.w

        #normalize direction and return ray
        direction = tm.normalize(direction)
        return Ray(self.eye_position, direction)
