typedef float2 cmplx;

cmplx cmul(cmplx a, cmplx b);

inline cmplx cmul(cmplx a, cmplx b)
{
  return (cmplx)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

/*
//multiply a * conj(b)
inline cmplx cmul_c(cmplx a, cmplx b)
{
  return (cmplx)(a.x*b.x + a.y*b.y, -a.x*b.y + a.y*b.x);
}
*/

/* NOTE: OpenCL scheduling is such that global_id(0) varies fastest,
i.e. for 2d array, xid=global_id(0) maps best to column index(!) of C-contiguous numpy buffer!
*/

__kernel
__attribute__((reqd_work_group_size(8,8,1)))
void
premul_field_2d(
    const unsigned int N1_0, //number of rows in field
    const unsigned int N1_1, //number of colums in field
	__constant cmplx* w1_0,
	__constant cmplx* w1_1,
	__global cmplx* field,
	__global cmplx* work)
{
  const unsigned int id0 = get_global_id(0); // column!
  const unsigned int id1 = get_global_id(1);
  const unsigned int workid = id1*get_global_size(0) + id0;
  const unsigned int fieldid = id1*N1_1 + id0;
  cmplx value;
  if (id0<N1_1 & id1<N1_0)
    {
      value = cmul( cmul(field[fieldid], w1_0[id1] ), w1_1[id0] );
    }
  else
    {
      value = (cmplx)(0, 0);
    }
  work[workid] = value;
}


__kernel void
multiply_workF_2d(
    __constant cmplx* w12F_0, // __attribute__((max_constant_size(4096))),
	__constant cmplx* w12F_1, // __attribute__((max_constant_size(4096))),
	__global cmplx* workF)
{
  const unsigned int id0 = get_global_id(0);  // column
  const unsigned int id1 = get_global_id(1);  // row
  const unsigned int Ncol = get_global_size(0);
  const unsigned int gid = id1*Ncol + id0;

  workF[gid] = cmul(cmul(workF[gid], w12F_0[id1]), w12F_1[id0]);
}


__kernel void
postmul_field_2d(
    const unsigned int N12p_0,
	const unsigned int N12p_1,
	__constant cmplx* w2_0,
	__constant cmplx* w2_1,
	float scale,
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
  field[fieldid] = scale * cmul(cmul(work[id1*N12p_1 + id0], w2_0[id1]), w2_1[id0]);

}
