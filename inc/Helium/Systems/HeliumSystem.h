#pragma once

class HeliumCore;

class HeliumSystem
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
