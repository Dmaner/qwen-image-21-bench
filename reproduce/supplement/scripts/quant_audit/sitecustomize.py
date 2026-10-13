"""Opt-in, worker-local C1 evidence; never used by headline or quality runs.

C1_QUANT_AUDIT=1 installs exact-source import hooks. The diagnostic reads tensor
metadata only: no reductions, transfers, synchronization, tensor/cache retention,
weight mutation, or inference-output changes. Actual GEMM execution needs trace.
"""
import os
import sys

C1_QUANT_AUDIT_INSTALLED = False
OMNI_SHA = "3dc35694b3d458fc1461a8fe71832ac12e8c8cea"
from pathlib import Path
import json
from audit_protocol import SCHEMA_VERSION, expected_names, inventory_errors
SOURCES = json.loads((Path(__file__).resolve().parent / "source_guards.json").read_text())
GUARDED = {}
FORWARD_CONTEXT = None
REQUEST_MARKER_CACHE = None


def _request_marker():
    """Read caller metadata only; absence means initialization/dummy execution."""
    import hashlib
    global REQUEST_MARKER_CACHE
    directory = os.environ.get('C1_AUDIT_OUT')
    if not directory:
        return None
    path = Path(directory) / 'request_start.json'
    try:
        stamp = path.stat().st_mtime_ns
    except FileNotFoundError:
        return None
    if REQUEST_MARKER_CACHE is not None and REQUEST_MARKER_CACHE[0] == stamp:
        return REQUEST_MARKER_CACHE[1]
    raw = path.read_bytes()
    marker = json.loads(raw)
    if marker.get('engine_initialization_complete') is not True:
        raise RuntimeError('Audit request marker must be written after engine initialization')
    ordinal = marker['request_ordinal']
    result = {'request_start': marker, 'request_start_sha256': hashlib.sha256(raw).hexdigest(),
              'request_start_file': f'request_start_{ordinal:04d}.json'}
    REQUEST_MARKER_CACHE = (stamp, result)
    return result



def _type_name(value):
    return None if value is None else type(value).__module__ + "." + type(value).__qualname__


def _tensor_meta(value):
    if value is None:
        return None
    import torch
    if not isinstance(value, torch.Tensor):
        return {"python_type": _type_name(value)}
    return {"dtype": str(value.dtype), "shape": list(value.shape), "device": str(value.device),
            "stride": list(value.stride()), "numel": value.numel(), "element_size": value.element_size(),
            "logical_nbytes": value.numel() * value.element_size(),
            "is_meta": value.is_meta}


def _write(kind, payload):
    import json
    from datetime import datetime, timezone
    from pathlib import Path
    directory = os.environ.get("C1_AUDIT_OUT")
    if not directory:
        raise RuntimeError("C1_AUDIT_OUT is required before loading the worker")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": SCHEMA_VERSION, "kind": kind, "time_utc": datetime.now(timezone.utc).isoformat(),
             "pid": os.getpid(), "ppid": os.getppid(), "arm": os.environ.get("C1_AUDIT_ARM"),
             "omni_sha": OMNI_SHA, "evidence_scope": "actual worker tensors, metadata only; separate diagnostic",
             "environment": {key: os.environ.get(key) for key in
                             ("RANK", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES", "VLLM_BATCH_INVARIANT")},
             **payload}
    target = out / f"{kind}_{os.getpid()}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def _guard(module):
    import hashlib
    from pathlib import Path
    actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    expected = SOURCES[module.__name__]
    if actual != expected:
        raise RuntimeError(f"C1 quant audit exact-source guard failed for {module.__name__}: {actual} != {expected}")
    GUARDED[module.__name__] = actual
    return actual


def _config_meta(config):
    if config is None:
        return None
    result = {"class": _type_name(config), "method": config.get_name()}
    for key in ("is_checkpoint_fp8_serialized", "activation_scheme", "ignored_layers", "ignored_layers_match_mode"):
        if hasattr(config, key):
            result[key] = getattr(config, key)
    if hasattr(config, "_components"):
        result["components"] = {key: _config_meta(value) for key, value in config._components.items()}
        result["default"] = _config_meta(config._default)
    if hasattr(config, "quantized_layers"):
        result["quantized_layers"] = {key: list(value) for key, value in config.quantized_layers.items()}
    return result


def _inventory(runner):
    import torch
    from vllm.model_executor.layers.linear import LinearBase
    pipeline = runner.pipeline
    if type(pipeline).__name__ != "QwenImage21Pipeline":
        # This pipeline's actual class name is checked below by the guarded module.
        if type(pipeline).__module__ != "vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21":
            raise RuntimeError("C1 audit only supports the pinned Qwen-Image 2.1 pipeline")
    _guard(sys.modules[type(pipeline).__module__])
    components = {}
    for name in ("transformer", "text_encoder", "vae"):
        component = getattr(pipeline, name)
        parameters = []
        dtype_numel = {}
        for param_name, tensor in component.named_parameters():
            meta = _tensor_meta(tensor)
            parameters.append({"name": name + "." + param_name, **meta})
            dtype_numel[meta["dtype"]] = dtype_numel.get(meta["dtype"], 0) + meta["numel"]
        components[name] = {"class": _type_name(component), "parameter_dtype_numel": dtype_numel,
                            "parameters": parameters,
                            "fp8_parameter_names": [x["name"] for x in parameters if "float8" in x["dtype"]]}
    linears = []
    for name, module in pipeline.named_modules():
        if not isinstance(module, (torch.nn.Linear, LinearBase)):
            continue
        method = getattr(module, "quant_method", None)
        linears.append({"name": name, "module_class": _type_name(module),
                        "quant_method": _type_name(method), "weight": _tensor_meta(getattr(module, "weight", None)),
                        "scales": {key: _tensor_meta(getattr(module, key, None)) for key in
                                   ("weight_scale", "weight_scale_inv", "input_scale", "input_scale_ub")},
                        "postprocess_flag": getattr(module, "_already_called_process_weights_after_loading", None),
                        "fp8_linear_kernel_class": _type_name(getattr(method, "fp8_linear", None)),
                        "use_marlin": getattr(method, "use_marlin", None),
                        "weight_quant_key": str(getattr(method, "weight_quant_key", None)),
                        "activation_quant_key": str(getattr(method, "activation_quant_key", None))})
    transformer = pipeline.transformer
    block_linears = [x for x in linears if x["name"].startswith("transformer.transformer_blocks.")]
    quantized = [x for x in block_linears if x["quant_method"] and "Fp8PerTensorOnlineLinearMethod" in x["quant_method"]]
    boundaries = [x for x in linears if x["name"].startswith("transformer.")
                  and not x["name"].startswith("transformer.transformer_blocks.")]
    arm = os.environ.get("C1_AUDIT_ARM")
    expected = expected_names(arm)
    od = runner.od_config
    te_config = pipeline.text_encoder.config
    import importlib.metadata
    payload = {"source_guards": dict(GUARDED),
               "model_architecture": {"dit_blocks": len(transformer.transformer_blocks),
                                      "te_language_layers": getattr(getattr(te_config, "text_config", te_config), "num_hidden_layers", None),
                                      "te_vision_blocks": getattr(getattr(te_config, "vision_config", None), "depth", None)},
               "adapter_source_sha256": _adapter_hash(),
               "audit_protocol_sha256": _protocol_hashes(),
               "packages": {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers", "diffusers")},
               "resolved_quantization": _config_meta(od.quantization_config),
               "transformer_quantization": _config_meta(transformer.quant_config),
               "resolved_runtime": {"dtype": str(od.dtype), "enforce_eager": od.enforce_eager,
                                    "enable_cuda_graph_decode": od.enable_cuda_graph_decode,
                                    "prefix_kv_cache_dtype": transformer.prefix_kv_cache_dtype,
                                    "diffusion_kv_cache_dtype": od.diffusion_kv_cache_dtype},
               "components": components, "linears": linears, "boundary_linears": boundaries,
               "expected_counts": {key: len(value) for key, value in expected.items()},
               "observed_fp8_dit_linear_count": len(quantized),
               "observed_fp8_te_linear_count": sum("Fp8PerTensorOnlineLinearMethod" in (x["quant_method"] or "")
                                                  for x in linears if x["name"].startswith("text_encoder.")),
               "kernel_evidence_boundary": "selected kernel class/weights are not an executed-GEMM trace"}
    payload["inventory_errors"] = inventory_errors({**payload, "omni_sha": OMNI_SHA}, arm)
    _write("worker_inventory", payload)
    if payload["inventory_errors"]:
        raise RuntimeError("C1 supplement worker inventory failed: " + "; ".join(payload["inventory_errors"][:8]))
    _capture_text_encoder(pipeline.text_encoder)
    # Read the actual Q/K/V supplied to each attention operation once. No tensor
    # references escape the callback. Capture both prefill and decode to observe
    # the actual attention input dtype after prefix dequantization as well.
    observations = {}
    first_request = None
    for index, block in enumerate(transformer.transformer_blocks):
        attention = block.attn.attn
        def capture(module, args, kwargs, *, block_index=index):
            nonlocal first_request
            request = _request_marker()
            if request is None:
                return
            if first_request is None:
                first_request = request
            if request['request_start_sha256'] != first_request['request_start_sha256']:
                return
            if FORWARD_CONTEXT is None:
                return
            identity = FORWARD_CONTEXT
            phase_observations = observations.setdefault(identity, {})
            if str(block_index) in phase_observations:
                return
            query = args[0] if len(args) > 0 else kwargs.get("query")
            key = args[1] if len(args) > 1 else kwargs.get("key")
            value = args[2] if len(args) > 2 else kwargs.get("value")
            phase_observations[str(block_index)] = {
                "query": _tensor_meta(query), "key": _tensor_meta(key), "value": _tensor_meta(value),
                "attention_class": _type_name(module),
                "backend": str(getattr(module, "attn_backend", None)),
                "backend_class": _type_name(getattr(module, "attn_backend", None)),
            }
            _write("attention_runtime", {**first_request, "observations": observations, "expected_blocks": len(transformer.transformer_blocks),
                                         "observation": "first actual attention call per block/branch/prefill-or-decode; metadata-only pre-hook",
                                         "execution_boundary": "input dtype/backend observed; GPU kernel must be confirmed by trace"})
        attention.register_forward_pre_hook(capture, with_kwargs=True)


def _capture_text_encoder(encoder):
    # Input/output metadata from actual calls. First call per Linear only;
    # emit after root forward so this proves language and (for edit) vision use.
    import torch
    from vllm.model_executor.layers.linear import LinearBase
    observations = {}
    first_request = None
    for local_name, layer in encoder.named_modules():
        if not isinstance(layer, (torch.nn.Linear, LinearBase)):
            continue
        holder = {}
        def capture(module, args, kwargs, output, *, name="text_encoder." + local_name, own=holder):
            nonlocal first_request
            request = _request_marker()
            if request is None:
                return
            if first_request is None:
                first_request = request
            if request['request_start_sha256'] != first_request['request_start_sha256']:
                return
            value = args[0] if args else kwargs.get("input", kwargs.get("x"))
            result = output[0] if isinstance(output, tuple) else output
            observations[name] = {"input": _tensor_meta(value), "output": _tensor_meta(result),
                                  "module_class": _type_name(module),
                                  "quant_method": _type_name(getattr(module, "quant_method", None))}
            own["handle"].remove()
            own.clear()
        holder["handle"] = layer.register_forward_hook(capture, with_kwargs=True)
    def flush(module, args, kwargs, output):
        request = _request_marker()
        if first_request is None or request is None or request['request_start_sha256'] != first_request['request_start_sha256']:
            return
        hidden = getattr(output, "hidden_states", None)
        _write("text_encoder_runtime", {**first_request, "linears": dict(observations),
                "last_hidden_state": _tensor_meta(hidden[-1]) if hidden else None,
                "vision_executed": any(name.startswith("text_encoder.model.visual.") for name in observations),
                "observation": "first actual Linear input/output per layer; emitted after successful encoder.forward",
                "execution_boundary": "dtype/method and module execution observed; FP8 GEMM kernel still requires trace"})
    encoder.register_forward_hook(flush, with_kwargs=True)


def _protocol_hashes():
    import hashlib
    return {name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
            for name in ("audit_protocol.py", "expected_topology.json", "source_guards.json")}


def _adapter_hash():
    import hashlib
    from pathlib import Path
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _patch_runner(module):
    import functools
    cls = module.DiffusionModelRunner
    original = cls.load_model
    @functools.wraps(original)
    def load_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        _inventory(self)
        return result
    cls.load_model = load_model


def _patch_transformer(module):
    import functools
    import inspect
    cls = module.QwenImage21Transformer2DModel
    original = cls.forward
    signature = inspect.signature(original)
    observed = {}
    first_request = None
    call_counts = {'initialization_without_marker': 0, 'first_real_request': 0, 'later_real_requests': 0}
    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        global FORWARD_CONTEXT
        nonlocal first_request
        request = _request_marker()
        if request is None:
            call_counts['initialization_without_marker'] += 1
            return original(self, *args, **kwargs)
        if first_request is None:
            first_request = request
        if request['request_start_sha256'] != first_request['request_start_sha256']:
            call_counts['later_real_requests'] += 1
            return original(self, *args, **kwargs)
        call_counts['first_real_request'] += 1
        bound = signature.bind(self, *args, **kwargs)
        cache = bound.arguments.get("kv_cache")
        branch = bound.arguments.get("cache_branch", "cond")
        ready = bool(cache and "key" in cache[0].get(branch, {}))
        phase = "decode" if ready else "prefill"
        identity = f"{branch}_{phase}"
        previous = FORWARD_CONTEXT
        FORWARD_CONTEXT = identity
        try:
            result = original(self, *args, **kwargs)
        finally:
            FORWARD_CONTEXT = previous
        if cache is not None and identity not in observed:
            blocks = []
            logical = {}
            storages = {}
            bf16_payload = 0
            for index, block_cache in enumerate(cache):
                tensors = block_cache.get(branch, {})
                tensor_metadata = {}
                for name, tensor in tensors.items():
                    meta = _tensor_meta(tensor)
                    # Storage APIs read allocator metadata only; no Tensor is
                    # copied, reduced, synchronized or retained by this hook.
                    storage = tensor.untyped_storage()
                    storage_key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
                    if storage_key not in storages:
                        storages[storage_key] = len(storages)
                    meta.update(storage_index=storages[storage_key], storage_nbytes=storage.nbytes())
                    tensor_metadata[name] = meta
                    logical[name] = logical.get(name, 0) + meta['logical_nbytes']
                    if name in ('key', 'value'):
                        bf16_payload += meta['numel'] * 2
                blocks.append({"block": index, "branch": branch, "tensors": tensor_metadata})
            # Only Python metadata is retained; no cache or Tensor object is saved.
            observed[identity] = {"phase": phase, "branch": branch, "blocks": blocks,
                                  "configured_prefix_dtype": self.prefix_kv_cache_dtype,
                                  "cache_bytes": {"logical_by_name": logical,
                                                  "logical_total": sum(logical.values()),
                                                  "unique_storage_total": sum(key[2] for key in storages),
                                                  "unique_storage_count": len(storages),
                                                  "bf16_equivalent_kv_payload": bf16_payload},
                                  "byte_scope": "one branch's persistent prefix tensors including scales, excluding allocator overhead and transient decode buffers; not peak VRAM"}
            _write("prefix_runtime", {**first_request, "observations": observed,
                                      "transformer_call_counts_at_capture": dict(call_counts),
                                      "observation": "after successful first caller request transformer.forward; first prefill/decode per branch; startup without marker excluded"})
        return result
    cls.forward = forward


if os.environ.get("C1_QUANT_AUDIT") == "1":
    import importlib.abc
    import importlib.machinery
    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in SOURCES:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                raise ImportError("Cannot load C1 audited module " + fullname)
            original_exec = spec.loader.exec_module
            def exec_module(module):
                original_exec(module)
                _guard(module)
                if fullname.endswith("diffusion_model_runner"):
                    _patch_runner(module)
                elif fullname.endswith("qwen_image_21_transformer"):
                    _patch_transformer(module)
            spec.loader.exec_module = exec_module
            return spec
    sys.meta_path.insert(0, _Finder())
    C1_QUANT_AUDIT_INSTALLED = True
