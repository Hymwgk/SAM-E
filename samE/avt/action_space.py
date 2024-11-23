import torch
import numpy as np  
from math import sin,cos,pi,acos,atan2
from scipy.spatial.transform import Rotation as R




class SpherCartTransform():
    """关注世界坐标系和局部球坐标之间的转换关系"""
    
    def __init__(self,spher_center=None,
                 local_rotation=None,
                 boundary_offset=True) -> None:
        
        self.wtl_pos = np.array(spher_center) if spher_center else np.zeros([3,])
        self.wtl_rot = R.from_quat(local_rotation).as_matrix() if local_rotation else np.eye(3)
        
        self.boundary_offset = boundary_offset
    
    def set_spher_center(self,spher_center):
        """设置局部球坐标系的中心"""
        self.wtl_pos = np.array(spher_center)
    
    def set_local_rotation(self,local_rotation):
        """设置local坐标系的旋转"""
        self.wtl_rot = R.from_quat(local_rotation).as_matrix()
            
    def _cart_to_spher(self,cartesion_position:np.ndarray,):
        """输出弧度"""
        # 目标点在local坐标系中的三维坐标
        ltp_trans = cartesion_position
        # 将ltc_pos笛卡尔坐标 转换为  local球坐标值
        local_r = np.linalg.norm(ltp_trans)
        local_theta = acos(ltp_trans[2]/local_r) 
        local_phi = atan2(ltp_trans[1],ltp_trans[0])
        
        if self.boundary_offset:
            if local_phi>np.pi:
                local_phi -= 2*np.pi

        #返回球坐标
        return np.array([local_r,local_theta,local_phi])


    def _spher_to_cart(self,local_spher_pos:np.ndarray,):
        """输入球坐标为弧度
        """
        local_r,local_theta,local_phi = tuple(local_spher_pos)

        # local_theta = np.radians(local_theta)
        # local_phi = np.radians(local_phi)
        
        if self.boundary_offset:
            # 将phi变回规范的[0,2pi]范围 
            if local_phi<0:
                local_phi += 2*np.pi
            elif local_phi > 2*np.pi:
                local_phi -= 2*np.pi 
        
        # 计算local笛卡尔坐标
        local_pos_x = local_r* sin(local_theta)*cos(local_phi)
        local_pos_y = local_r* sin(local_theta)*sin(local_phi)
        local_pos_z = local_r* cos(local_theta)
        ltp_trans = np.array([local_pos_x,local_pos_y,local_pos_z])
        
        return ltp_trans
    
    def get_world_cart(self,local_spher_pos:np.ndarray):
        """局部球坐标转世界坐标"""
        ltp_trans = self._spher_to_cart(local_spher_pos)
        # 计算wtp
        wtp_trans = self.wtl_rot.dot(ltp_trans) + self.wtl_pos
        # 
        return wtp_trans
        

    def get_local_spher(self,world_cart_pos:np.ndarray):
        """将世界坐标系中的笛卡尔坐标，转换到局部球坐标系中的球坐标"""
        wtp_trans = world_cart_pos
        # 目标点在 local 视点笛卡尔坐标系下位置
        ltp_trans = self.wtl_rot.T.dot(wtp_trans) - self.wtl_rot.T.dot(self.wtl_pos)
        local_spher = self._cart_to_spher(ltp_trans)
        return local_spher

class ViewpointSpace():
    def __init__(self,
                 spher_bound, # deg
                 cart_bound,
                 spher_center=None,
                 local_rotation=None,
                 boundary_offset=True) -> None:
        
        self.coord_tsf = SpherCartTransform(spher_center=spher_center,
                                            local_rotation=local_rotation,
                                            boundary_offset=boundary_offset)
        self.spher_bound = spher_bound
        self.cart_bound = cart_bound

    def boundary_check(self,boundary):
        pass
    
    def rad2deg(self, rad_angle: np.ndarray) -> np.ndarray:
        return np.degrees(rad_angle)

    def deg2rad(self, deg_angle: np.ndarray) -> np.ndarray:
        return np.radians(deg_angle)        
            
        
    def get_cart_coord(self,spher_coord: np.ndarray) -> np.ndarray:
        """计算世界笛卡尔坐标，输入为角度"""
        cart_pos = self.coord_tsf.get_world_cart(local_spher_pos=self.deg2rad(spher_coord))
        #cart_pos = np.clip(cart_pos,self.cart_bound[:3],self.cart_bound[3:])
    
        return np.clip(cart_pos,self.cart_bound[:3],self.cart_bound[3:])
    
    def get_spher_coord(self,cart_coord:np.ndarray) -> np.ndarray:
        """视点的世界笛卡尔坐标，转换为局部球坐标"""
        spher_coord = self.coord_tsf.get_local_spher(world_cart_pos=cart_coord)
        return np.clip(self.rad2deg(spher_coord),self.spher_bound[:3],self.spher_bound[3:])
    
class DisViewpointSpace(ViewpointSpace):
    def __init__(self, 
                 cart_bound, 
                 rotation_bound,
                 spher_bound=[1.3,20,-135, 1.3,60,135], 
                 spher_res=[0.05,5.0,5.0],
                 rotation_res=5.0,
                 spher_center=None, 
                 local_rotation=None, 
                 boundary_offset=True) -> None:
        super().__init__(spher_bound, cart_bound, spher_center, local_rotation, boundary_offset)
        
        self.spher_res = np.array(spher_res)
        self.rotation_res = rotation_res
        self.rotation_bound = rotation_bound
        #self.sph
        # viewpoint球坐标最大索引
        self.max_vp_spher_indices = (spher_bound[3:]-spher_bound[:3])//spher_res - 1
        # 旋转姿态的最大索引
        self.max_vp_rotation_indices = (rotation_bound[:3]-rotation_bound[3:])//rotation_res - 1
    
    def _clip_shper_indices(self,spher_indices:np.ndarray) -> np.ndarray:
        return np.clip(spher_indices,np.zeros([3]),self.max_vp_spher_indices)
    
    def _clip_rotation_indices(self,rotation_indices:np.ndarray) -> np.ndarray:
        return np.clip(rotation_indices,np.zeros([3]),self.max_vp_spher_indices)
        
    
    def _indices2disc_shper(self,spher_indices:np.ndarray) -> np.ndarray:
        """将球坐标索引转换为离散球坐标"""
        cliped_spher_indices = self._clip_shper_indices(spher_indices)
        # 将indices转换为离散网格的中间位置
        disc_spher_coord = cliped_spher_indices * self.spher_res + self.spher_bound[:3] + 0.5 * self.spher_res
        
        return disc_spher_coord
    
    def _indices2disc_rotation(self,rotation_indices:np.ndarray) -> np.ndarray:
        """将旋转索引转换为离散的旋转动作"""
        cliped_rotation_indices = self._clip_rotation_indices(rotation_indices)
        disc_rotation = cliped_rotation_indices * self.rotation_res + self.rotation_bound + 0.5 * self.rotation_res
        
        return disc_rotation
    
    def get_env_action(self,model_output:np.ndarray) -> np.ndarray:
        """将模型输出的离散动作索引，转换为场景动作，或者叫索引"""        
        vp_spher_indeces = model_output[:3]
        vp_rotation_indices = model_output[3:]
        # 计算离散球坐标
        disc_spher_coord = self._indices2disc_shper(vp_spher_indeces)
        # 计算对应的世界坐标
        world_cart_coord = self.get_cart_coord(disc_spher_coord)
        # 计算离散的旋转姿态
        disc_rotation = self._indices2disc_rotation(vp_rotation_indices)
        
        return np.concatenate([world_cart_coord,disc_rotation],axis=-1)
    
    
    def get_action_indices(self,env_action:np.ndarray) -> np.ndarray:
        """将视点的环境动作转变为离散索引"""
        world_cart_coord = env_action[:3]
        vp_rotation = env_action[3:]
        
        # 先计算连续的球坐标
        vp_spher_coord = self.get_spher_coord(world_cart_coord)
        # 转换为离散球坐标索引，并限幅
        vp_spher_indices = self._clip_shper_indices((vp_spher_coord - self.spher_bound[:3])//self.spher_res)
        # 转换为离散旋转索引，并限幅
        vp_rotation_indices = self._clip_rotation_indices((vp_rotation - self.rotation_bound[:3])//self.rotation_res)
        
        return np.concatenate([vp_spher_indices,vp_rotation_indices],axis=-1)
        
        

        
        
    