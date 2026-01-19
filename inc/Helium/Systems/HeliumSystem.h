#pragma once

#include <Helium/HeliumCommon.h>

class HeliumCore;

class HELIUM_API HeliumSystem
{
public:
    HeliumSystem(HeliumCore* core) : core(core) {}
    virtual ~HeliumSystem() = default;

    virtual void Initialize() {}
    virtual void Update(float dt) {}
    virtual void Render() {}
    virtual void Shutdown() {}

protected:
    HeliumCore* core = nullptr;
};
