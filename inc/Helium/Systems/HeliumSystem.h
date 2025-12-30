#pragma once

class HeliumCore;

class HeliumSystem
{
public:
    HeliumSystem(HeliumCore* core) : m_Core(core) {}
    virtual ~HeliumSystem() = default;

    virtual void Initialize() {}
    virtual void Update(float dt) {}
    virtual void Render() {}
    virtual void Shutdown() {}

protected:
    HeliumCore* m_Core = nullptr;
};
