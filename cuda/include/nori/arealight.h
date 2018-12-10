//
// Created by tholugo on 24.11.17.
//

#ifndef NORIGL_AREALIGHT_H
#define NORIGL_AREALIGHT_H

/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Romain Prévost

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
#include <cmath>
#include <nori/emitter.h>
#include <nori/warp.h>
#include <nori/shape.h>
#include <nori/frame.h>
#include <iostream>
#include <cuda_runtime_api.h>

NORI_NAMESPACE_BEGIN

class AreaEmitter : public Emitter {
    public:
       __host__ virtual size_t getSize() const override {return sizeof(AreaEmitter);};
       __host__ AreaEmitter(const PropertyList &props);
       __host__ virtual std::string toString() const override ;
       __device__  Color3f eval(const EmitterQueryRecord & lRec) const ;
       __device__  Color3f sample(EmitterQueryRecord & lRec, const Point2f & sample) const ;
       __device__  float pdf(const EmitterQueryRecord &lRec) const  ;
    protected:
        Color3f m_radiance;
    };


NORI_NAMESPACE_END
#endif //NORIGL_AREALIGHT_H
