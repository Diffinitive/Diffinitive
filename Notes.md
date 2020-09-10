# Notes

## Known size of range and domain?
It might be a good idea let tensormappings know the size of their range and domain as a constant. This probably can't be enforced on the abstract type but maybe we should write our difference operators this way. Having this as default should clean up the thinking around adjoints of boundary operators. It could also simplify getting high performance out of repeated application of regioned TensorMappings.