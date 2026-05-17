#ifndef RDX4_STGS_H
#define RDX4_STGS_H

extern void r4_stage_q1(float*, int32_t);

extern void r4_stage_q4(float*, int32_t, const float*);

extern void r4_stage(float*, int32_t, size_t, const float*, size_t);

#endif