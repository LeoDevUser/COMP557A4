import geometry as geom
from helperclasses import Ray, Intersection
from camera import Camera

import numpy as np

import taichi as ti
import taichi.math as tm

shadow_epsilon = 10**(-2)

@ti.data_oriented
class Scene:
    def __init__(self,
                 jitter: bool,
                 samples: int,
                 camera: Camera,
                 ambient: tm.vec3,
                 lights: ti.template(),
                 nb_lights: int,
                 spheres: ti.template(),
                 nb_spheres: int,
                 planes: ti.template(),
                 nb_planes: int,
                 aaboxes: ti.template(),
                 nb_aaboxes: int,
                 meshes: ti.template(),
                 nb_meshes: int,
                 meshes_verts: np.array,
                 meshes_faces: np.array,
                 ):
        self.jitter = jitter  # should rays be jittered
        self.samples = samples  # number of rays per pixel
        self.camera = camera
        self.ambient = ambient  # ambient lighting
        self.lights = lights  # all lights in the scene
        self.nb_lights = nb_lights
        self.spheres = spheres
        self.planes = planes
        self.aaboxes = aaboxes
        self.meshes = meshes
        self.nb_spheres = nb_spheres
        self.nb_planes = nb_planes
        self.nb_aaboxes = nb_aaboxes
        self.nb_meshes = nb_meshes

        self.meshes_verts = ti.Vector.field(3, shape=(max(1, meshes_verts.shape[0])), dtype=float)
        self.meshes_verts.from_numpy(meshes_verts)
        self.meshes_faces = ti.Vector.field(3, shape=(max(1, meshes_faces.shape[0])), dtype=int)
        self.meshes_faces.from_numpy(meshes_faces)

        self.image = ti.Vector.field( n=3, dtype=float, shape=(self.camera.width, self.camera.height) )

        self.offsets = ti.field(dtype=ti.f32, shape=((self.samples - 1) * (self.samples - 1) + 1, 2))


    @ti.kernel
    def render( self, iteration_count: int ):
        for x,y in ti.ndrange(self.camera.width, self.camera.height):
            if (y == x) and x%10 == 0: print(".",end='')
            ray = self.camera.create_ray( x, y, self.jitter )
            intersect = self.intersect_scene(ray, 0, float('inf'))
            sample_colour = tm.vec3(0, 0, 0) # background colour
            if intersect.is_hit:
                sample_colour = self.compute_shading(intersect, ray)
            self.image[x,y] += (sample_colour - self.image[x,y]) / iteration_count
        print() # end of line after one dot per 10 rows


    @ti.func
    def intersect_scene(self, ray: Ray, t_min: float, t_max: float) -> Intersection:
        best = Intersection() # default is no intersection (is_hit = False)        
        ti.loop_config(serialize=True) 
        for i in range(self.nb_spheres):
            hit = geom.intersectSphere(self.spheres[i], ray, t_min, t_max )
            if hit.is_hit: best = hit; t_max = hit.t # keep best hit only
        ti.loop_config(serialize=True) 
        for i in range(self.nb_planes):
            hit = geom.intersectPlane(self.planes[i], ray, t_min, t_max )
            if hit.is_hit: best = hit; t_max = hit.t                
        ti.loop_config(serialize=True) 
        for i in range(self.nb_aaboxes):
            hit = geom.intersectAABox(self.aaboxes[i], ray, t_min, t_max )
            if hit.is_hit: best = hit; t_max = hit.t
        ti.loop_config(serialize=True) 
        for i in range(self.nb_meshes):
            hit = geom.intersectMesh(self.meshes[i], self.meshes_verts, self.meshes_faces, ray, t_min, t_max)
            if hit.is_hit: best = hit; t_max = hit.t
        return best

    @ti.func
    def compute_shading(self, intersect: Intersection, ray: Ray) -> tm.vec3:
        sample_colour = tm.vec3(0, 0, 0)

        # Ambient shading
        sample_colour += self.ambient * intersect.mat.diffuse

        ti.loop_config(serialize=True) 
        for l in range(self.nb_lights):

            light = self.lights[l]
        
            light_dir = tm.vec3(0, 0, 0)
            attenuation = 0.0
            distance_to_light = 0.0

            if light.ltype == 0:
                # Directional light: vector is already normalized direction toward light
                light_dir = light.vector
                attenuation = 1.0
                distance_to_light = float('inf')
            else:
                # Point light: vector is position, compute direction
                light_vector = light.vector - intersect.position
                distance_to_light = tm.length(light_vector)
                light_dir = light_vector.normalized()
                
                kq = light.attenuation.x
                kl = light.attenuation.y
                kc = light.attenuation.z
                attenuation = 1.0 / (kc + kl * distance_to_light + kq * distance_to_light * distance_to_light)
            
            normal = intersect.normal.normalized()
            view_dir = (-ray.direction).normalized()
            
            diffuse_factor = tm.max(0.0, normal.dot(light_dir))
            diffuse = intersect.mat.diffuse * light.colour * diffuse_factor * attenuation
            
            specular = tm.vec3(0, 0, 0)
            if tm.length(intersect.mat.specular) > 0.0:
                half_vector = (view_dir + light_dir).normalized()
                spec_factor = tm.max(0.0, normal.dot(half_vector))
                spec_factor = tm.pow(spec_factor, intersect.mat.shininess.x)
                specular = intersect.mat.specular * light.colour * spec_factor * attenuation
            
            # TODO: Objective 6: Implement shadow rays
            
            #check for shadows if the surface is facing the light
            if diffuse_factor > 0.0:
                #create shadow ray from surface point toward light
                shadow_ray = Ray(intersect.position, light_dir)
                
                shadow_hit = self.intersect_scene(shadow_ray, shadow_epsilon, distance_to_light)
                
                if not shadow_hit.is_hit:
                    sample_colour += diffuse + specular

        return sample_colour
