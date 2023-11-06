#pragma once
#pragma once
#pragma once
// going to need an update function
#include <Jolt/Jolt.h>


// Jolt includes
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>
#include <iostream>
#include "Physics.h"

// STL includes
class Rigidbody : public Component
{
public:
	void Init() {

	};

	Rigidbody(const JPH::Shape *inShape = new JPH::SphereShape(.5), JPH::RVec3Arg inPosition = JPH::RVec3Arg(0.0f,0.0f,0.0f), JPH::QuatArg inRotation = Quat::sIdentity(), JPH::EMotionType inMotionType = EMotionType::Dynamic, JPH::ObjectLayer inObjectLayer = Layers::MOVING) {
	
		JPH::BodyCreationSettings shape_settings(inShape, inPosition, inRotation,inMotionType,inObjectLayer);
		m_id = Physics::GetInstance()->GetPhysicsSystem()->GetBodyInterface().CreateAndAddBody(shape_settings, EActivation::Activate);
	};
	
	void AddLinearVelocity(Vec3 velocity) {
		Physics::GetInstance()->GetPhysicsSystem()->GetBodyInterface().AddLinearVelocity(m_id, velocity);
	}



	JPH::Vec3 GetPosition() {
		return Physics::GetInstance()->GetPhysicsSystem()->GetBodyInterface().GetCenterOfMassPosition(m_id);
	}
	void SetPosition(JPH::Vec3 position) {
		Physics::GetInstance()->GetPhysicsSystem()->GetBodyInterface().SetPosition(m_id, position, EActivation::Activate);
	}
	
	JPH::Vec3 GetVelocity() {
		return Physics::GetInstance()->GetPhysicsSystem()->GetBodyInterface().GetLinearVelocity(m_id);
	}

	void SetVelocity(JPH::Vec3 velocity) {
		Physics::GetInstance()->GetPhysicsSystem()->GetBodyInterface().SetLinearVelocity(m_id, velocity);
	}

	float GetFriction() {
		return Physics::GetInstance()->GetPhysicsSystem()->GetBodyInterface().GetFriction(m_id);
	}

	void SetFriction(float friction) {
		Physics::GetInstance()->GetPhysicsSystem()->GetBodyInterface().SetFriction(m_id, friction);
	}


	float GetBounce() {
		return Physics::GetInstance()->GetPhysicsSystem()->GetBodyInterface().GetRestitution(m_id);
	}

	void SetBounce(float bounce) {
		Physics::GetInstance()->GetPhysicsSystem()->GetBodyInterface().SetRestitution(m_id, bounce);
	}


	



	


	JPH::BodyID m_id;
private: 
};