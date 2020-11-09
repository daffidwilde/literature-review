This is where your application's keys should be stored for the Springer and
IEEE APIs. The instructions for getting and registering a key are available in
the Arcas
[documentation](https://arcas.readthedocs.io/en/latest/Guides/api_key.html).

Store each key in a file here, named after the API, like so:

```python
# keys/ieee.py

api_key = "My IEEE API key"
```

Then, you can set your keys for the current `conda` environment's `arcas`
installation by running the script `src/set_api_keys.py`.

If you are not using the `conda` environment, follow the instructions in the
documentation linked above to set the keys manually.
