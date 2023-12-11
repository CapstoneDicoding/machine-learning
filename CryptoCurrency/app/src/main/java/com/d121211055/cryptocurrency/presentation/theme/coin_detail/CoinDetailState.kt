package com.d121211055.cryptocurrency.presentation.theme.coin_detail

import com.d121211055.cryptocurrency.domain.model.Coin
import com.d121211055.cryptocurrency.domain.model.CoinDetail

data class CoinDetailState(
    val isLoading: Boolean = false,
    val coin: CoinDetail? = null,
    val error: String = ""
)
