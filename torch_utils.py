if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def autocast(dtype=torch.float16):
    if device == "mps":
        return torch.autocast("mps", dtype=dtype)
    elif device == "cuda":
        return torch.autocast("cuda", dtype=dtype)
    else:
        return torch.autocast("cpu", dtype=dtype)

def synchronize():
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    else:
        pass

def empty_cache():
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    else:
        pass
