from typing import Dict, List
import jax.numpy as jnp

def timeseries_statistics(stats_list: List, mode: str = "append") -> Dict:
    """Combines a list of turbulent statistics 
    - in a single statistic with the time series data (mode = append)
    or 
    - in a single statistic with time-averaged data (mode = mean)

    :param stats_list: _description_
    :type stats_list: List
    :param mode: _description_, defaults to "append"
    :type mode: str, optional
    :raises NotImplementedError: _description_
    :return: _description_
    :rtype: Dict
    """
    assert mode in ["append", "mean"]
    timeseries_stats = {}
    
    stat_0 = stats_list[0]
    for key, value in stat_0.items():
        timeseries_stats[key] = {}
        for subkey in value.keys():
            timeseries_stats[key][subkey] = []
            for stat in stats_list:
                value = stat[key][subkey]
                timeseries_stats[key][subkey].append(value)
                
            if mode == "append":
                timeseries_stats[key][subkey] = jnp.array(
                    timeseries_stats[key][subkey])
            elif mode == "mean":
                timeseries_stats[key][subkey] = jnp.mean(
                    jnp.array(timeseries_stats[key][subkey]), axis=0
                )
            else:
                raise NotImplementedError

    return timeseries_stats