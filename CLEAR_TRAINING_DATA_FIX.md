# Fix for Clear Training Data Error

## Issue

```
TypeError: 'bool' object is not callable
File "src/clear_training_data.py", line 156, in clear_all_training_data
    cache_result = clear_caches()
```

## Root Cause

The parameter name `clear_caches` (boolean) was shadowing the function name `clear_caches()`. When Python tried to call `clear_caches()` inside `clear_all_training_data()`, it thought `clear_caches` referred to the boolean parameter instead of the function.

## Fix Applied

Renamed the parameter from `clear_caches` to `clear_caches_flag` to avoid name shadowing:
- Function signature: `def clear_all_training_data(..., clear_caches_flag: bool = True, ...)`
- Inside function: `if clear_caches_flag: cache_result = clear_caches()  # Now correctly refers to function`
- API call updated: `clear_all_training_data(..., clear_caches_flag=True, ...)`

## Files Modified

1. `src/clear_training_data.py` - Renamed parameter to avoid shadowing
2. `src/api_server.py` - Updated call site to use new parameter name

## Status

✅ **Fixed** - Parameter renamed to avoid name collision

