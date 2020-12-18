typedef float2 cmplx;

cmplx cmul(cmplx a, cmplx b);

inline cmplx cmul(cmplx a, cmplx b)
{
  return (cmplx)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

//multiply a * conj(b)
inline cmplx cmul_c(cmplx a, cmplx b)
{
  return (cmplx)(a.x*b.x + a.y*b.y, -a.x*b.y + a.y*b.x);
}

/* NOTE: OpenCL scheduling is such that global_id(0) varies fastest,
i.e. for 2d array, xid=global_id(0) maps best to column index(!) of C-contiguous numpy buffer!
*/

__kernel
__attribute__((reqd_work_group_size(8,8,1)))
void
premul_field_2d(
    const unsigned int N1_0, //number of rows in field
    const unsigned int N1_1, //number of colums in field
	__global cmplx* field,
	__global cmplx* work)
{
  const unsigned int id0 = get_global_id(0); // column!
  const unsigned int id1 = get_global_id(1);
  const unsigned int workid = id1*get_global_size(0) + id0;
  const unsigned int fieldid = id1*N1_1 + id0;
  cmplx value;
  if ((id0<(N1_1)) & (id1<(N1_0)))
    {
      value = field[fieldid];//field[fieldid];
    }
  else
    {
      value = (cmplx)(0.f, 0.f);
    }
  work[workid] = value;
}


__kernel void
multiply_workF_2d(
                  __global cmplx* workF,
                  __global cmplx* G
                  )
{
  const unsigned int id0 = get_global_id(0);  // column
  const unsigned int id1 = get_global_id(1);  // row
  const unsigned int Ncol = get_global_size(0);
  const unsigned int gid = id1*Ncol + id0;
  const unsigned int gid2 = id0 * get_global_size(1) + id1;  
  workF[gid] =  cmul(workF[gid], G[gid2]); // maybe stupid?
  
}

__kernel void
postmul_field_2d(
    const unsigned int N12p_0,
	const unsigned int N12p_1,
    const unsigned int N1_0,
    const unsigned int N1_1,
    __global cmplx* work,
	__global cmplx* field)
{

  const unsigned int id0 = get_global_id(0);  // column
  const unsigned int id1 = get_global_id(1);
  const unsigned int fieldid = id1*get_global_size(0)+ id0;
  const unsigned int workid = id1*N12p_1 + id0 + N12p_1 * N1_0 + N1_1;
  cmplx value;
  
  value = work[workid];


  field[fieldid] = value;
}


__kernel void
postmul_field_2d_FR(
    const unsigned int N12p_0,
	const unsigned int N12p_1,
    __global cmplx* work,
	__global cmplx* field)
{
  /*
  extract global_size() subarray of work with size global_size to field, multiplied with w2 and scaling
  */
  // schneller
  const unsigned int id0 = get_global_id(0);  // column
  const unsigned int id1 = get_global_id(1);
  const unsigned int fieldid = id1*get_global_size(0) + id0;

  field[fieldid] = work[fieldid];

}