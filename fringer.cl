__constant sampler_t sampler_params = (CLK_NORMALIZED_COORDS_TRUE |
                                       CLK_ADDRESS_CLAMP_TO_EDGE |
                                       CLK_FILTER_LINEAR);
__constant sampler_t sampler_E = (CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR);

//#define PHI_MIN 0.f
//#define PHI_MAX 2.f

#define N_MIN 1.f
#define N_MAX 2.f

#define X_MIN -10.f
#define X_MAX  10.f

#define PIXELSIZE 15.f //pixel size in micron

/*
inline float smooth(float x_normalized, int shift, float4 params, __read_only image2d_t table_E) //
{
    const float x_0 = params.s0 + shift*PIXELSIZE;
    const float xi_p = params.s1;
    const float xi_m = params.s2;
    
    const float c = 1.f/(xi_m + xi_p);
    const float x = PIXELSIZE*x_normalized - x_0;
    
    return
    //x<0 ? c * xi_m *  exp(x/xi_m) :
    //      (1.f - c * xi_p*exp(-x/xi_p)); 
    
    
    x<0 ? c * xi_m * 0.45f * exp(x/xi_m) + 0.55f * c * xi_m * (erf(x/xi_m) + 1.f) :
          0.45f * (1.f - c * xi_p*exp(-x/xi_p)) + 0.55f * c * (xi_m+xi_p*erf(x/xi_p));   
}
*/

inline float smooth(float x_normalized, int shift, float4 params, __read_only image2d_t table_E) //
{
    const float x_0 = params.s0 + shift*PIXELSIZE;
    const float xi_p = params.s1;
    const float xi_m = params.s2;
    const float n = params.s3;

    const float c = 2.f/((xi_m + xi_p));
    const float x = PIXELSIZE*x_normalized - x_0;

    float y = 0;
    if (x<0)
      {
	const float2 Epos = (float2)((n - N_MIN)/(N_MAX-N_MIN), (x/xi_m - X_MIN)/(X_MAX-X_MIN) );
	y = c * xi_m * read_imagef(table_E, sampler_E, Epos).x;
      }
    else
      {
	const float2 Epos = (float2)((n - N_MIN)/(N_MAX-N_MIN), (x/xi_p - X_MIN)/(X_MAX-X_MIN) );
	y = c * ( xi_p * read_imagef(table_E, sampler_E, Epos).x + .5f * (xi_m - xi_p));
      }
    
    return y;
}


inline float4 get_params(__read_only image2d_t table_params, float ph1, float ph2)
{
    float2 p = (float2)((ph1-PHI_MIN)/(PHI_MAX-PHI_MIN),
                      (ph2-PHI_MIN)/(PHI_MAX-PHI_MIN));
    
    return read_imagef(table_params, sampler_params, p);

    //if (ph1<ph2){return (float4)(-1.05f, 4.443f, 3.733f, 0.f); // x0, x_p, x_m, 0.f}                   
    //else
    //{return (float4)(2.008f, 3.071f, 3.517f, 0.f);}

}

__kernel void
fringe1d(
    __global const float *phase,
    __global float *phase_smooth,
    __read_only image2d_t table_params,
    __read_only image2d_t table_E
)
{
    int pixel_id = get_group_id(0);
    int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);
    
    const float x_local = (lid + 0*0.5f)*(1.f/get_local_size(0));
    
    const int k_0 = pixel_id;
    const int k_m = clamp(pixel_id-1,0,(int)get_num_groups(0)-1);
    const int k_p = clamp(pixel_id+1,0,(int)get_num_groups(0)-1);

    const float ph_m = phase[k_m];
    const float ph_0 = phase[k_0];
    const float ph_p = phase[k_p];

    float phi = ph_0 + (ph_p-ph_0) *  smooth(x_local, 1, get_params(table_params, ph_0, ph_p), table_E)
      + (ph_0-ph_m) * (smooth(x_local, 0, get_params(table_params, ph_m, ph_0), table_E) - 1.f);

    phase_smooth[gid] =  phi;
}


__kernel void
fringe1d_grad(
    __global float *phase_grad, //output
    __global const float *phase,
    __global const float *phase_smooth_grad,
    __read_only image2d_t table_params,
    __read_only image2d_t table_E
)
{
    __local float storage[16];

    int pixel_id = get_group_id(0);
    unsigned int lid = get_local_id(0);
    
    const float x_local = (lid + 0*0.5f)*(1.f/get_local_size(0)); // 0.5?????????
        
    const int k_0 = pixel_id;
    const int k_m = clamp(pixel_id-1,0,(int)get_num_groups(0)-1);
    const int k_p = clamp(pixel_id+1,0,(int)get_num_groups(0)-1);

    const float ph_m = phase[k_m];
    const float ph_0 = phase[k_0];
    const float ph_p = phase[k_p];

    float phsgrad_m = phase_smooth_grad[k_m * get_local_size(0) + lid];
    float phsgrad_0 = phase_smooth_grad[k_0 * get_local_size(0) + lid];
    float phsgrad_p = phase_smooth_grad[k_p * get_local_size(0) + lid];

    storage[lid] = (
        phsgrad_0
        + (phsgrad_0 - phsgrad_m) * (smooth(x_local, 0, get_params(table_params, ph_m, ph_0), table_E) - 1.f)   
        + (phsgrad_p - phsgrad_0) *  smooth(x_local, 1, get_params(table_params, ph_0, ph_p), table_E)
    );

    //reduction
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int stride = get_local_size(0)/2; stride>0; stride >>= 1)
    {
        if (lid<stride)
        {
            storage[lid] += storage[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        phase_grad[pixel_id] = storage[0];
    }
}


inline float4 mymix(float4 v0, float4 v1, float x)
{
    x = 1.5f*(x-0.5f) + 0.5f;  // magic constant 1.5
    float a = smoothstep(1.f, 0.f, x);
    return v0*a + (1.f-a)*v1;
}



#define GET_phase_clamped(i,j) (phase[clamp(grpid0+i, 0, (int)get_num_groups(0)-1) + stride_grp * clamp(grpid1+j, 0, (int)get_num_groups(1)-1)]) 
#define PH(i,j) (storage_phase[(i+1) + 3*(j+1)])

__kernel void
fringe2dV2(
    __global const float *phase, //input, lores
    __global float *phase_smooth, //input, hires
    __read_only image2d_t table_params_x,
    __read_only image2d_t table_params_y,
    __read_only image2d_t table_E,
    int debug
    )
{
    const int gid0 = get_global_id(0); // hires idx
    const int gid1 = get_global_id(1);
    const int gid = gid0 + gid1*get_global_size(0); //linear index hires
    
    const int grpid0 = get_group_id(0); //lores idx
    const int grpid1 = get_group_id(1);
    const int stride_grp = get_num_groups(0);        

    const int lid0 = get_local_id(0);
    const int lid1 = get_local_id(1);    
    __local float storage_phase[9]; //local memory cache for phase neighbouring pixels lores
    
    
    //preload neighbours phase once per workgroup
    if ((lid0<3) && (lid1<3))
    {
        storage_phase[lid0 + lid1*3] = GET_phase_clamped(lid0-1, lid1-1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //cached read lores pixel values neighbours
    float v00 = PH( 0, 0);
    float vp0 = PH( 1, 0);    
    float vm0 = PH(-1, 0);
    float v0p = PH( 0, 1);
    float v0m = PH( 0,-1);
    float vpp = PH( 1, 1);
    float vmp = PH(-1, 1);
    float vpm = PH( 1,-1);
    float vmm = PH(-1,-1);

    __local float4 local_par[12]; //local memory cache for transition parameters
    
    if ((lid0<2) && (lid1<3))
    {
        local_par[    3*lid0 + lid1] = get_params(table_params_x, PH(lid0-1, lid1-1), PH(lid0, lid1-1));
        local_par[6 + 3*lid0 + lid1] = get_params(table_params_y, PH(lid1-1, lid0-1), PH(lid1-1, lid0));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    const float4 par_mm_0m = local_par[0]; //get_params(table_params_x, PH(-1,-1), PH( 0,-1));
    const float4 par_m0_00 = local_par[1]; //get_params(table_params_x, PH(-1, 0), PH( 0, 0));
    const float4 par_mp_0p = local_par[2]; //get_params(table_params_x, PH(-1,+1), PH( 0,+1));

    const float4 par_0m_pm = local_par[3]; //get_params(table_params_x, PH( 0,-1), PH(+1,-1));
    const float4 par_00_p0 = local_par[4]; //get_params(table_params_x, PH( 0, 0), PH(+1, 0));
    const float4 par_0p_pp = local_par[5]; //get_params(table_params_x, PH( 0,+1), PH(+1,+1));

    const float4 par_mm_m0 = local_par[6]; //get_params(table_params_y, PH(-1,-1), PH(-1, 0));
    const float4 par_0m_00 = local_par[7]; //get_params(table_params_y, PH( 0,-1), PH( 0, 0));
    const float4 par_pm_p0 = local_par[8]; //get_params(table_params_y, PH(+1,-1), PH(+1, 0));

    const float4 par_m0_mp = local_par[9]; //get_params(table_params_y, PH(-1, 0), PH(-1,+1));
    const float4 par_00_0p = local_par[10]; //get_params(table_params_y, PH( 0, 0), PH( 0,+1));
    const float4 par_p0_pp = local_par[11]; //get_params(table_params_y, PH(+1, 0), PH(+1,+1));
    
    
    const float x_local = lid0 * (1.f/get_local_size(0));
    const float y_local = lid1 * (1.f/get_local_size(1));

    const float4 Ipar_00_p0 = (y_local < 0.5f) ?
            mymix(par_00_p0, par_0m_pm, 0.5f-y_local) :
            mymix(par_00_p0, par_0p_pp, -(0.5f - y_local));

    const float4 Ipar_m0_00 = (y_local < 0.5f) ?
            mymix(par_m0_00, par_mm_0m, 0.5f-y_local) :
            mymix(par_m0_00, par_mp_0p, -(0.5f-y_local));

    const float4 Ipar_00_0p = (x_local < 0.5f) ?
            mymix(par_00_0p, par_m0_mp, 0.5f-x_local) :
            mymix(par_00_0p, par_p0_pp, -(0.5f-x_local));

    const float4 Ipar_0m_00 = (x_local < 0.5f) ?
            mymix(par_0m_00, par_mm_m0, 0.5f-x_local) : 
            mymix(par_0m_00, par_pm_p0, -(0.5f-x_local));

    const float s00_p0 =  smooth(x_local, 1, Ipar_00_p0, table_E);
    const float sm0_00 = (smooth(x_local, 0, Ipar_m0_00, table_E) - 1.f);
    const float s00_0p =  smooth(y_local, 1, Ipar_00_0p, table_E);
    const float s0m_00 = (smooth(y_local, 0, Ipar_0m_00, table_E) - 1.f);

    float value = 0;
    
    value = v00
      + (vp0-v00) * s00_p0
      + (v00-vm0) * sm0_00
      + (v0p-v00) * s00_0p
      + (v00-v0m) * s0m_00
      + (vpp - vp0 - v0p + v00) * (s00_p0 * s00_0p)
      + (vp0 - vpm - v00 + v0m) * (s0m_00 * s00_p0)
      + (v0p - vmp - v00 + vm0) * (sm0_00 * s00_0p)
      + (v00 - vm0 - v0m + vmm) * (sm0_00 * s0m_00)
    ; 

    phase_smooth[gid] = value;
    
}

__kernel void
fringe2dV2_grad(
    __global float *grad_phase, //output, lores
    __global const float *phase, //input, lores
    __global const float *grad_phase_smooth, //input, hires
    __read_only image2d_t table_params_x,
    __read_only image2d_t table_params_y,
    __read_only image2d_t table_E,
    int debug)
{
    //const int gid0 = get_global_id(0); // hires idx
    //const int gid1 = get_global_id(1);
    //const int gid = gid0 + gid1*get_global_size(0); //linear index hires
    
    const int grpid0 = get_group_id(0); //lores idx
    const int grpid1 = get_group_id(1);
    const int stride_grp = get_num_groups(0);

    const int lid0 = get_local_id(0);
    const int lid1 = get_local_id(1);
    const unsigned int lid = lid0 + lid1*get_local_size(0);
    __local float storage_phase[9]; //local memory cache for phase neighbouring pixels lores
        
    //preload neighbours phase once per workgroup
    if ((lid0<3) && (lid1<3))
    {
        storage_phase[lid0 + lid1*3] = GET_phase_clamped(lid0-1, lid1-1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local float4 local_par[12]; //local memory cache for transition parameters
    
    if ((lid0<2) && (lid1<3))
    {
        local_par[    3*lid0 + lid1] = get_params(table_params_x, PH(lid0-1, lid1-1), PH(lid0, lid1-1));
        local_par[6 + 3*lid0 + lid1] = get_params(table_params_y, PH(lid1-1, lid0-1), PH(lid1-1, lid0));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    const float4 par_mm_0m = local_par[0]; //get_params(table_params_x, PH(-1,-1), PH( 0,-1));
    const float4 par_m0_00 = local_par[1]; //get_params(table_params_x, PH(-1, 0), PH( 0, 0));
    const float4 par_mp_0p = local_par[2]; //get_params(table_params_x, PH(-1,+1), PH( 0,+1));

    const float4 par_0m_pm = local_par[3]; //get_params(table_params_x, PH( 0,-1), PH(+1,-1));
    const float4 par_00_p0 = local_par[4]; //get_params(table_params_x, PH( 0, 0), PH(+1, 0));
    const float4 par_0p_pp = local_par[5]; //get_params(table_params_x, PH( 0,+1), PH(+1,+1));

    const float4 par_mm_m0 = local_par[6]; //get_params(table_params_y, PH(-1,-1), PH(-1, 0));
    const float4 par_0m_00 = local_par[7]; //get_params(table_params_y, PH( 0,-1), PH( 0, 0));
    const float4 par_pm_p0 = local_par[8]; //get_params(table_params_y, PH(+1,-1), PH(+1, 0));

    const float4 par_m0_mp = local_par[9]; //get_params(table_params_y, PH(-1, 0), PH(-1,+1));
    const float4 par_00_0p = local_par[10]; //get_params(table_params_y, PH( 0, 0), PH( 0,+1));
    const float4 par_p0_pp = local_par[11]; //get_params(table_params_y, PH(+1, 0), PH(+1,+1));
    
    
    const float x_local = lid0 * (1.f/get_local_size(0));
    const float y_local = lid1 * (1.f/get_local_size(1));

    const float4 Ipar_00_p0 = (y_local < 0.5f) ?
            mymix(par_00_p0, par_0m_pm, 0.5f-y_local) :
            mymix(par_00_p0, par_0p_pp, -(0.5f - y_local));

    const float4 Ipar_m0_00 = (y_local < 0.5f) ?
            mymix(par_m0_00, par_mm_0m, 0.5f-y_local) :
            mymix(par_m0_00, par_mp_0p, -(0.5f-y_local));

    const float4 Ipar_00_0p = (x_local < 0.5f) ?
            mymix(par_00_0p, par_m0_mp, 0.5f-x_local) :
            mymix(par_00_0p, par_p0_pp, -(0.5f-x_local));

    const float4 Ipar_0m_00 = (x_local < 0.5f) ?
            mymix(par_0m_00, par_mm_m0, 0.5f-x_local) : 
            mymix(par_0m_00, par_pm_p0, -(0.5f-x_local));

    const float s00_p0 =  smooth(x_local, 1, Ipar_00_p0, table_E);
    const float sm0_00 = (smooth(x_local, 0, Ipar_m0_00, table_E) - 1.f);
    const float s00_0p =  smooth(y_local, 1, Ipar_00_0p, table_E);
    const float s0m_00 = (smooth(y_local, 0, Ipar_0m_00, table_E) - 1.f);

#define G(i,j) \
    grad_phase_smooth[ \
      clamp( grpid0 + i, 0, (int)get_num_groups(0)-1) * get_local_size(0) + lid0 + \
      (clamp( grpid1 + j, 0, (int)get_num_groups(1)-1) * get_local_size(1) + lid1)*get_global_size(0) \
		       ]
    
    float g00 = G( 0, 0);
    float gp0 = G( 1, 0);    
    float gm0 = G(-1, 0);
    float g0p = G( 0, 1);
    float g0m = G( 0,-1);
    float gpp = G( 1, 1);
    float gmp = G(-1, 1);
    float gpm = G( 1,-1);
    float gmm = G(-1,-1);

    local float storage[16*16]; //TODO: local size
    
    storage[lid]  = g00
      + (gp0 - g00) * s00_p0
      + (g00 - gm0) * sm0_00
      + (g0p - g00) * s00_0p
      + (g00 - g0m) * s0m_00
      + (gpp - gp0 - g0p + g00) * (s00_p0 * s00_0p)
      + (gp0 - gpm - g00 + g0m) * (s0m_00 * s00_p0)
      + (g0p - gmp - g00 + gm0) * (sm0_00 * s00_0p)
      + (g00 - gm0 - g0m + gmm) * (sm0_00 * s0m_00)
    ;

    //reduction
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int stride = get_local_size(1)*get_local_size(0)/2; stride>0; stride >>= 1)
      {
	if (lid<stride)
	  {
	    storage[lid] += storage[lid + stride];
	  }
	barrier(CLK_LOCAL_MEM_FENCE);
      }

    if (lid==0)
      {
	grad_phase[grpid0 + grpid1*stride_grp] = storage[0];
      }
}
