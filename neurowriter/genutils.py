
# coding: utf-8

# Utilities for data generation
#
# @author Álvaro Barbero Jiménez



def maskedgenerator(generatorfunction):
    """Decorator that adds outputs masking to a generator.
    
    A "mask" parameter is added to the generator function, which expects
    a list of boolean variables. The mask is iterated in parallel to the
    generator, blocking from the output those items with a False value
    in the mask. If the mask is depleted it is re-cycled.
    """
    def mskgenerator(*args, **kwargs):
        if "mask" in kwargs:
            mask = kwargs["mask"]
            del kwargs["mask"]
        else:
            mask = [True]
        for i, item in enumerate(generatorfunction(*args, **kwargs)):
            if mask[i % len(mask)]:
                yield item
                
    return mskgenerator
