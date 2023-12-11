package com.d121211055.cryptocurrency.domain.repository

import com.d121211055.cryptocurrency.data.remote.dto.CoinDetailDto
import com.d121211055.cryptocurrency.data.remote.dto.CoinDto

interface CoinRepository {
    suspend fun getCoins(): List<CoinDto>
    suspend fun getCoinById(coinId:String):CoinDetailDto
}